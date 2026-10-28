@isdefined(chebyshev_test_matrix) || include("chebyshev_helpers.jl")

@testset "Chebyshev eigensolver" begin
    H = chebyshev_test_matrix(6)
    E = eigvals(Hermitian(Matrix(H)))
    op = Hamster.HermitianCSCOperator(H)
    proto = Matrix{ComplexF64}(undef, 0, 0)

    # Block products agree with the sparse matrix
    X = randn(ComplexF64, 5, size(H, 1)); Z = randn(ComplexF64, 5, size(H, 1)); Y = similar(X)
    Hamster.block_matvec!(Y, op, X, 2.0, 0.5, -1.5, Z, 0.25, X)
    @test Y ≈ 2 .* (transpose(H * transpose(X)) .- 0.5 .* X) .- 1.5 .* Z .+ 0.25 .* X

    # In the gap, inside a band and at the lower edge of the spectrum
    for target in [0.0, 1.2, -4.0], Neig in [6, 20]
        Es, V, info = Hamster.chebyshev_eigen(op, proto, size(H, 1), Neig, target=target)
        @test Es ≈ closest_eigenvalues(E, Neig, target) atol=1e-8
        @test V * V' ≈ I atol=1e-8
        @test norm(transpose(H * transpose(V)) .- Es .* V) < 1e-7
        @test info.iterations > 0
    end

    # Degenerate levels are returned with their multiplicity
    Hd = chebyshev_test_matrix(5, kramers=true)
    Ed = eigvals(Hermitian(Matrix(Hd)))
    Es, _, _ = Hamster.chebyshev_eigen(Hamster.HermitianCSCOperator(Hd), proto, size(Hd, 1), 8, target=0.1)
    @test Es ≈ closest_eigenvalues(Ed, 8, 0.1) atol=1e-8

    # Warm start from the eigenvectors of a slightly different matrix
    H2 = H + 0.01 * chebyshev_test_matrix(6, seed=2)
    E2 = eigvals(Hermitian(Matrix(H2)))
    op2 = Hamster.HermitianCSCOperator(H2)
    _, V, _ = Hamster.chebyshev_eigen(op, proto, size(H, 1), 10, target=0.0)
    Es_cold, _, cold = Hamster.chebyshev_eigen(op2, proto, size(H, 1), 10, target=0.0)
    Es_warm, _, warm = Hamster.chebyshev_eigen(op2, proto, size(H, 1), 10, target=0.0, X0=V)
    @test Es_cold ≈ closest_eigenvalues(E2, 10, 0.0) atol=1e-8
    @test Es_warm ≈ Es_cold atol=1e-8
    @test warm.matvecs < cold.matvecs

    # Sparse diagonalization interface, including the dense fallback for small matrices
    Es, vs = diagonalize(H, Neig=6, target=0.0, method="chebyshev")
    @test Es ≈ closest_eigenvalues(E, 6, 0.0) atol=1e-8
    @test size(vs) == (size(H, 1), 6)
    Hs = chebyshev_test_matrix(2)
    Es, vs = diagonalize(Hs, Neig=4, target=0.0, method="chebyshev")
    @test Es ≈ closest_eigenvalues(eigvals(Hermitian(Matrix(Hs))), 4, 0.0)
    @test_throws ErrorException diagonalize(H, Neig=6, target=0.0, method="lobpcg")
end
