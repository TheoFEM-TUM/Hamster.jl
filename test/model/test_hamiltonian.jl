@testset "Hamiltonian" begin
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
