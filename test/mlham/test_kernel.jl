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

    kernel = HamiltonianKernel(params, data_points, 0.2, [], false)
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

    # Test expand mode
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

@testset "CsPbBr3 Hermitian Hamiltonian" begin
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

    kernel = HamiltonianKernel([strc], [basis], model, comm, conf)
    Hr = get_hr(kernel, Hamster.Dense(), 1)
    Hk = get_hamiltonian(Hr, strc.Rs, zeros(3, 1))
    @test abs(sum(Hk[1] .- Hermitian(Hk[1]))) < 1e-7

    dR = kernel.structure_descriptors[1]
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
@testset "Settings in the header of the ML parameter file" begin
    mktempdir() do dir
        file = joinpath(dir, "ml_fit")
        fitted = get_empty_config()
        for (key, value) in (("rcut", 5.5), ("sim_params", 0.2), ("strc_scale", 0.7), ("apply_distance_distortion", true),
                             ("key_dims", "1 3"))
            set_value!(fitted, key, "ML", value)
        end
        kernel = HamiltonianKernel([1.0, 2.0], [SVector(0.1, 0.2), SVector(0.3, 0.4)], 0.2, [], false)
        write_params(kernel, fitted, filename=file)
        header = Hamster.read_ml_header(file)
        @test Set(keys(header)) == Set(Hamster.ML_HEADER_TAGS)
        @test header["key_dims"] == "1 3" && header["strc_scale"] == "0.7"

        # Settings not in the ML block are taken from the file
        conf = get_empty_config()
        set_value!(conf, "init_params", "ML", file * ".dat")
        Hamster.apply_ml_settings!(conf)
        @test Hamster.get_ml_sim_params(conf) == 0.2 && Hamster.get_ml_strc_scale(conf) == 0.7
        @test Hamster.get_ml_apply_distance_distortion(conf) && Hamster.get_ml_key_dims(conf) == [1, 3]
        @test Hamster.read_ml_params(conf, filename=file)[1] == [1.0, 2.0]

        # Settings in the ML block must agree (up to rounding)
        set_value!(conf, "sim_params", "ML", 0.2 + 1e-14)
        Hamster.apply_ml_settings!(conf)
        set_value!(conf, "sim_params", "ML", 0.3)
        @test_throws r"fitted with sim_params = 0.2, but sim_params = 0.3" Hamster.apply_ml_settings!(conf)
        @test_throws r"fitted with sim_params = 0.2" Hamster.read_ml_params(conf, filename=file)

        # Files with the header of earlier versions keep the defaults for the missing settings
        write(file * "_old.dat", "begin unknown\n  rcut = 5.5\n  sim_params = 0.2\n  env_scale = 1.0\n" *
                                 "  apply_distortion = false\n  apply_orthogonality = false\nend\n\n1.0 0.1 0.2\n")
        old = get_empty_config()
        set_value!(old, "init_params", "ML", file * "_old.dat")
        Hamster.apply_ml_settings!(old)
        @test Hamster.get_ml_strc_scale(old) == 1.0 && Hamster.get_ml_sim_params(old) == 0.2
        @test Hamster.read_ml_params(old, filename=file * "_old")[1] == [1.0]
    end
end
