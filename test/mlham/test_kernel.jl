@testset "HamiltonianKernel Tests" begin
    # Define test inputs
    ws = [0.5, 1.0, 1.5]
    xs = [[1.0], [2.0], [3.0]]
    σ = 0.1
    
    # Test 1: test exp_sim function
    @test Hamster.exp_sim([1.0], [1.0], σ=0.1) ≈ 1.0
    @test Hamster.exp_sim([1.0], [2.0], σ=0.1) < 1.0
    @test Hamster.exp_sim([1.0], [3.0], σ=0.1) < Hamster.exp_sim([1.0], [2.0], σ=0.1)
end