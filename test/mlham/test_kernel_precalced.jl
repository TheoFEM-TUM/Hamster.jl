@testset "HamiltonianKernelPrecalced smoke" begin
    point = SVector{8, Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    second_point = SVector{8, Float64}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    descriptor = sparse([1, 1], [1, 2], [point, second_point], 2, 2)
    conf = get_empty_config()
    set_value!(conf, "verbosity", 0)
    base_kernel = HamiltonianKernel([2.0, 3.0], [point, second_point], 0.5, [[descriptor]], true)
    kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf)

    @test Hamster.get_params(kernel) == [2.0, 3.0]
    @test kernel.kp.datapoints == [point, second_point]
    @test kernel.kp.key_ranges == Dict((0,) => 1:2)
    @test kernel.sm.sim_params == 0.5

    expected = [2.0 * Hamster.exp_sim2(point, point, σ=0.5) + 3.0 * Hamster.exp_sim2(second_point, point, σ=0.5),
                2.0 * Hamster.exp_sim2(point, second_point, σ=0.5) + 3.0 * Hamster.exp_sim2(second_point, second_point, σ=0.5)]
    hr_dense = get_hr(kernel, Hamster.Dense(), 1)
    hr_sparse = get_hr(kernel, Hamster.Sparse(), 1)
    expected_matrix = [expected[1] expected[2]; 0.0 0.0]
    @test hr_dense[1] ≈ expected_matrix
    @test Matrix(hr_sparse[1]) ≈ expected_matrix

    set_params!(kernel, [4.0, 5.0])
    @test get_params(kernel) == [4.0, 5.0]
    copy_params!(kernel, Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf))
    @test get_params(kernel) == [2.0, 3.0]

    bad_kp = Hamster.Kernelpoints(
        [point, point],
        Int64[1, 1],
        [(0,), (1,)],
        Dict((0,) => 1:1, (1,) => 1:1),
        Dict((0,) => 1, (1,) => 1),
    )
    with_logger(NullLogger()) do
        @test !Hamster.verify_kernelpoints(bad_kp, get_empty_config(); key_dims=Int64[1])
    end
end

@testset "HamiltonianKernelPrecalced Tests" begin
    # Define test inputs
    ws = [0.5, 1.0, 1.5]
    xs = [[1.0], [2.0], [3.0]]
    σ = 0.1

    # Test 1: test exp_sim function
    @test Hamster.exp_sim([1.0], [1.0], σ=0.1) ≈ 1.0
    @test Hamster.exp_sim([1.0], [2.0], σ=0.1) < 1.0
    @test Hamster.exp_sim([1.0], [3.0], σ=0.1) < Hamster.exp_sim([1.0], [2.0], σ=0.1)
end

@testset "Precalced ML Parameter" begin
    conf = get_empty_config()
    set_value!(conf, "rcut", "ML", 5.0)
    set_value!(conf, "sim_params", "ML", 0.2)
    set_value!(conf, "env_scale", "ML", 0.8)
    set_value!(conf, "apply_distortion", "ML", false)

    N = 10
    params = rand(10)
    data_points = [SVector{8, Float64}(rand(8)) for i in 1:N]

    base_kernel = HamiltonianKernel(params, data_points, 0.2, [], false)
    kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf)
    write_params(kernel, conf)

    # Test 1: test parameter read and write
    params_1, data_points_1 = Hamster.read_ml_params(conf)
    @test params_1 == params
    @test data_points_1 == data_points

    # Test 2: test parameter initialization
    params_2, data_points_2 = Hamster.init_ml_params!(data_points, conf)
    @test all(x->x==0, params_2)
    @test data_points_2 == data_points

    set_value!(conf, "init_params", "ML", "ones")
    params_3, data_points_3 = Hamster.init_ml_params!(data_points, conf)
    @test all(x->x==1, params_3)
    @test data_points_3 == data_points

    set_value!(conf, "init_params", "ML", "random")
    params_4, data_points_4 = Hamster.init_ml_params!(data_points, conf)
    @test std(params_4) > 0
    @test data_points_4 == data_points

    data_points_dummy = [SVector{8, Float64}(rand(8)) for i in 1:N+1]
    set_value!(conf, "init_params", "ML", "ml_params.dat")
    params_5, data_points_5 = Hamster.init_ml_params!(data_points_dummy, conf)
    @test params_5 == params
    @test data_points_5 == data_points

    set_value!(conf, "mode", "ML", "expand")
    set_value!(conf, "init_params", "ML", "zeros")
    params_6, data_points_6 = Hamster.init_ml_params!(data_points_dummy, conf)
    @test length(params_6) == 21
    @test length(data_points_6) == 21
    @test params_6[1:10] == params
    @test params_6[11:21] == zeros(11)
    @test data_points_6[1:10] == data_points
    @test data_points_6[11:21] == data_points_dummy

    rm("ml_params.dat")
end

@testset "CsPbBr3 Hermitian Hamiltonian (precalced)" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_CsPbBr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "init_params", "ML", "rand")
    set_value!(conf, "init_params", "zeros")
    set_value!(conf, "verbosity", 0)
    strc = Structure(conf)
    basis = Basis(strc, conf)
    model = TBModel([strc], [basis], comm, conf)

    base_kernel = HamiltonianKernel([strc], [basis], model, comm, conf)
    kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf)
    Hr = get_hr(kernel, Hamster.Dense(), 1)
    Hk = get_hamiltonian(Hr, strc.Rs, zeros(3, 1))
    @test abs(sum(Hk[1] .- Hermitian(Hk[1]))) < 1e-7

    dR = base_kernel.structure_descriptors[1]
    dishermitian = Bool[]
    for R in axes(strc.Rs, 2)
        R⁻ = findfirst(R⃗-> R⃗ == -strc.Rs[:, R], eachcol(strc.Rs))
        for i in 1:14, j in i:14
            Δd = sum(abs.(dR[R][i, j] .- dR[R⁻][j, i]))
            push!(dishermitian, Δd < 1e-5)
            if Δd > 1e-5
                println("$i $j $(dR[R][i, j])")
                println("$i $j $(dR[R⁻][j, i])")
                @show dR[R][i, j] .- dR[R⁻][j, i]
                println("---")
            end
        end
    end
    @test all(dishermitian)
end
