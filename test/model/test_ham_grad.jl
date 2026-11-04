"""
    loss_gradient_loops(dL_dE, vs, Rs, ks) -> Array{ComplexF64, 3}

Gradient of the loss w.r.t. the real-space Hamiltonian, `dL_dHr[i, j, R]`, from the eigenvalue gradient
`dE[m, k]/dHr[R][i, j] = conj(vs[i, m, k]) exp(2πik⋅R) vs[j, m, k]` with explicit loops.
"""
function loss_gradient_loops(dL_dE, vs, Rs, ks)
    Nε = size(vs, 1)
    phase = Hamster.exp_2πi(Rs, ks)
    dL_dHr = zeros(ComplexF64, Nε, Nε, size(Rs, 2))
    for k in axes(ks, 2), m in axes(vs, 2), R in axes(Rs, 2), j in 1:Nε, i in 1:Nε
        dL_dHr[i, j, R] += dL_dE[m, k] * conj(vs[i, m, k]) * phase[R, k] * vs[j, m, k]
    end
    return dL_dHr
end

@testset "Loss gradient w.r.t. Hr" begin
    Hr = [Matrix(sprand(6, 6, 0.5)) for _ in 1:5]
    Rs = 2 .* rand(3, 5) .- 1
    ks = rand(3, 4) .- 0.5
    vs = nothing
    with_logger(Base.NullLogger()) do
        vs = diagonalize(get_hamiltonian(Hr, Rs, ks))[2]
    end
    dL_dE = rand(size(vs, 2), size(ks, 2))
    reference = loss_gradient_loops(dL_dE, vs, Rs, ks)

    # Dense
    direct = Hamster.get_loss_gradient_hr(dL_dE, vs, Rs, ks, Hamster.Dense())
    @test all(R -> direct[R] ≈ reference[:, :, R], axes(Rs, 2))
    @test direct[1] isa Matrix{ComplexF64}

    # Sparse: the same at the entries of the sparse iterator, zero elsewhere
    sp_iterator = [[(i, j) for i in 1:6 for j in 1:6 if (i + j + R) % 3 ≠ 0] for R in axes(Rs, 2)]
    masked(iterator) = [[(i, j) ∈ iterator[R] ? reference[i, j, R] : 0.0im for i in 1:6, j in 1:6] for R in axes(Rs, 2)]
    direct = Hamster.get_loss_gradient_hr(dL_dE, vs, Rs, ks, Hamster.Sparse(), sp_iterator)
    @test all(Matrix.(direct) .≈ masked(sp_iterator))
    @test direct[1] isa SparseMatrixCSC{ComplexF64, Int64}

    # Very uneven numbers of entries per lattice vector (all at R = 1, none at R = 3) and no entries at all
    sp_uneven = [R == 1 ? [(i, j) for i in 1:6 for j in 1:6] : R == 3 ? Tuple{Int64, Int64}[] : [(R, R)] for R in axes(Rs, 2)]
    direct = Hamster.get_loss_gradient_hr(dL_dE, vs, Rs, ks, Hamster.Sparse(), sp_uneven)
    @test all(Matrix.(direct) .≈ masked(sp_uneven))
    @test nnz(direct[3]) == 0
    empty = [Tuple{Int64, Int64}[] for _ in axes(Rs, 2)]
    @test all(R -> nnz(R) == 0, Hamster.get_loss_gradient_hr(dL_dE, vs, Rs, ks, Hamster.Sparse(), empty))
end
