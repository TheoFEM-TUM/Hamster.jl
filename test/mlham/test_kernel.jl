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

@testset "ML Parameter" begin
    conf = get_empty_config()
    set_value!(conf, "rcut", "ML", 5.0)
    set_value!(conf, "sim_params", "ML", 0.2)
    set_value!(conf, "env_scale", "ML", 0.8)
    set_value!(conf, "apply_distortion", "ML", false)

    N = 10
    params = rand(10)
    data_points = [SVector{8, Float64}(rand(8)) for i in 1:N]

    kernel = HamiltonianKernel(params, data_points, 0.2, [])
    write_params(kernel, conf)

    # Test 1: test parameter read and write
    params_1, data_points_1 = Hamster.read_ml_params(conf)
    @test params_1 == params
    @test data_points_1 == data_points

    # Test 2: test parameter initialization
    # set_value!(conf, "init_params", "ML", "zeros") [default]
    params_2, data_points_2 = Hamster.init_ml_params!(data_points, conf)
    @test all(x->x==0, params_2)
    @test data_points_2 == data_points

    set_value!(conf, "init_params", "ML", "ones")
    params_3, data_points_3 = Hamster.init_ml_params!(data_points, conf)
    @test all(x->x==1, params_3)
    @test data_points_3== data_points

    set_value!(conf, "init_params", "ML", "random")
    params_4, data_points_4 = Hamster.init_ml_params!(data_points, conf)
    @test std(params_4) > 0
    @test data_points_4 == data_points

    data_points_dummy = [SVector{8, Float64}(rand(8)) for i in 1:N+1]
    set_value!(conf, "init_params", "ML", "ml_params.dat")
    params_5, data_points_5 = Hamster.init_ml_params!(data_points_dummy, conf)
    @test params_5 == params
    @test data_points_5 == data_points

    rm("ml_params.dat")
end