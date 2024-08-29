@testset "Phase factor" begin
    # Test arrays
    R⃗ = rand(3)
    k⃗ = rand(3)
    result = Hamster.exp_2πi(R⃗, k⃗)
    expected = exp(2π * im * dot(R⃗, k⃗))
    @test isapprox(result, expected, rtol=1e-10)
    @test typeof(result) == ComplexF64

    # Test matrices
    Rs = rand(3, 5)
    ks = rand(3, 7)
    result = Hamster.exp_2πi(Rs, ks)
    @test size(result) == (5, 7)

    # Test matrix-vector
    Rs = rand(3)
    ks = rand(3, 7)
    result = Hamster.exp_2πi(Rs, ks)
    @test size(result) == (1, 7)

    Rs = rand(3, 5)
    ks = rand(3)
    result = Hamster.exp_2πi(Rs, ks)
    @test size(result) == (5,)
end

@testset "Hamiltonian" begin
    N = 32; NR = 25; Nk = 40
    Rs = rand(3, NR); ks = rand(3, Nk)

    Hr_dense = [rand(N, N) for R in axes(Rs, 2)]
    Hr_sp = [sprand(N, N, 0.01) for R in axes(Rs, 2)]

    # Test empty Hamiltonians
    Hk_empty = Hamster.get_empty_hamiltonians(N, Nk)
    @test length(Hk_empty) == Nk
    @test size(Hk_empty[1]) == (N, N)
    @test typeof(Hk_empty[1]) == Matrix{ComplexF64}

    Hk_empty = Hamster.get_empty_hamiltonians(N, Nk, sp_mode=true)
    @test length(Hk_empty) == Nk
    @test size(Hk_empty[1]) == (N, N)
    @test typeof(Hk_empty[1]) == SparseMatrixCSC{ComplexF64, Int64}

    # Test get_hamiltonian
    Hk = get_hamiltonian(Hr_sp, Rs, ks)
    # Sparse Hr should be converted to dense Hk for sp_mode=false
    @test typeof(Hk[1]) == Matrix{ComplexF64}

    Hk = get_hamiltonian(Hr_sp, Rs, ks, sp_mode=true)
    # Sparse Hr should lead to sparse Hk for sp_mode=true
    @test typeof(Hk[1]) == SparseMatrixCSC{ComplexF64, Int64}
    
    
end
