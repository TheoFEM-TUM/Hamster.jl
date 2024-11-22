"""
    get_eigenvalue_gradient(vs::Array{ComplexF64, 3}, Rs::Matrix{Float64}, ks::Matrix{Float64}) -> Array{Float64, 5}

Computes the gradient of energy eigenvalues with respect to the Hamiltonian matrix elements by performing explicit loops.
"""
function get_eigenvalue_gradient(vs::Array{ComplexF64, 3}, Rs::Matrix{Float64}, ks::Matrix{Float64})
    Nε = size(vs, 1); NR = size(Rs, 2); Nk = size(ks, 2)
    exp_2πiRk = Hamster.exp_2πi(Rs, ks)
    dε = zeros(Nε, Nε, NR, Nε, Nk)
    @tasks for k in 1:Nk
        for m in 1:Nε, R in 1:NR, i in 1:Nε, j in 1:Nε
            @views dε[i, j, R, m, k] = real(sum(@. conj(vs[i, m, k]) * exp_2πiRk[R, k] * vs[j, m, k]))
        end
    end
    return dε
end

@testset "Eigenvalue gradient" begin
    # Test 1: test that the gradient dE_dHr is the same as with explicit loops
    Hr_1 = [Hermitian(rand(4, 4)) for _ in 1:5]
    Rs = 2 .* rand(3, 5) .- 1
    ks = rand(3, 3) .- 0.5

    vs = diagonalize(get_hamiltonian(Hr_1, Rs, ks))[2]
    vs_ = Hamster.reshape_and_sparsify_eigenvectors(vs, Hamster.Dense())
    dE_dHr_an = Hamster.get_eigenvalue_gradient(vs_, Rs, ks) #[R, m, k][i, j]
    dE_dHr_old = get_eigenvalue_gradient(vs, Rs, ks) # i, j, R, m, k

    same_as_old = Bool[]
    for R in axes(dE_dHr_old, 3), j in axes(dE_dHr_old, 2), i in axes(dE_dHr_old, 1)
        for k in axes(dE_dHr_old, 5), m in axes(dE_dHr_old, 4)
            push!(same_as_old, dE_dHr_an[R, m, k][i, j] ≈ dE_dHr_old[i, j, R, m, k])
        end
    end
    @test all(same_as_old)

    # Test 2: test that the gradient dE_dHr is the same as with explicit loops for sparse arrays
    vs = diagonalize(get_hamiltonian(Hr_1, Rs, ks))[2]
    vs_ = Hamster.reshape_and_sparsify_eigenvectors(vs, Hamster.Sparse())
    dE_dHr_sp = Hamster.get_eigenvalue_gradient(vs_, Rs, ks)

    same_as_old = Bool[]
    for R in axes(dE_dHr_old, 3), j in axes(dE_dHr_old, 2), i in axes(dE_dHr_old, 1)
        for k in axes(dE_dHr_old, 5), m in axes(dE_dHr_old, 4)
            push!(same_as_old, dE_dHr_sp[R, m, k][i, j] ≈ dE_dHr_old[i, j, R, m, k])
        end
    end
    @test all(same_as_old)
    @test dE_dHr_sp[1, 1, 1] isa SparseMatrixCSC{Float64, Int64}

    # Test 3: test chain rule
    dL_dE = rand(size(dE_dHr_an, 2), size(ks, 2))
    dL_dHr = Hamster.chain_rule(dL_dE, dE_dHr_an, Hamster.Dense())
    @tensor dL_dHr_old[i, j, R] := dL_dE[m, k] * dE_dHr_old[i, j, R, m, k]
    same_as_old_2 = map(eachindex(dL_dHr)) do R
        dL_dHr[R] ≈ dL_dHr_old[:, :, R]
    end
    @test all(same_as_old_2)

    # Test 4: test chain rule for sparse matrices
    dL_dE = rand(size(dE_dHr_an, 2), size(ks, 2))
    dL_dHr = Hamster.chain_rule(dL_dE, dE_dHr_sp, Hamster.Sparse())
    @tensor dL_dHr_old[i, j, R] := dL_dE[m, k] * dE_dHr_old[i, j, R, m, k]
    same_as_old_2 = map(eachindex(dL_dHr)) do R
        dL_dHr[R] ≈ dL_dHr_old[:, :, R]
    end
    @test all(same_as_old_2)
end