gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"

@testset "TB Model GaAs" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "NNaxes", "As", true)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", string(@__DIR__)*"/test_files/rllm_true.dat")
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "rcut", 5)

    strc = Structure(conf, poscar_path=gaas_poscar)
    basis = Basis(strc, conf)
    model = TBModel(strc, basis, conf, initas=string(@__DIR__)*"/test_files/params.dat")
    Hr = get_hr(model, model.params)
    ks = read_from_file(string(@__DIR__)*"/test_files/kpoints.dat")
    Hk = get_hamiltonian(Hr, strc.Rs, ks)
    Es, vs = diagonalize(Hk)

    Es_correct = read_from_file(string(@__DIR__)*"/test_files/Es.dat")
    @test mean(abs.(Es .- Es_correct)) < 0.002
end

@testset "Test model gradient" begin
    # Test dense implementation
    dL_dHr = [rand(4, 4) for _ in 1:5]
    dL_dHr_t = cat(dL_dHr..., dims=3)
    hs_t = rand(5, 4, 4, 5)
    hs = Matrix{Matrix{Float64}}(undef, 5, 5)
    for v in 1:5, R in 1:5
        hs[v, R] = hs_t[v, :, :, R]
    end
    @tensor dV_t[v] := hs_t[v, i, j, R] * dL_dHr_t[i, j, R]
    dV = Hamster.get_model_gradient(hs, dL_dHr)
    @test dV ≈ dV_t

    # Test sparse implementation
    dL_dHr = [sparse(dL_dHr[i]) for i in 1:5]
    hs = Matrix{SparseMatrixCSC{Float64, Int64}}(undef, 5, 5)
    for v in 1:5, R in 1:5
        hs[v, R] = sparse(hs_t[v, :, :, R])
    end
    dV = Hamster.get_model_gradient(hs, dL_dHr)
    @test dV ≈ dV_t
end

@testset "Test init_params" begin
    # Test 1: Test parameter initialization
    model = TBModel(nothing, zeros(3), zeros(3), [zeros(Int64, 3)], [true, true, true])
    basis = nothing
    Hamster.init_params!(model, initas="ones")
    @test model.params == ones(3)
    @test get_params(model) == ones(3)
    Hamster.init_params!(model, initas="random")
    @test all(0 .< model.params .< 1)

    # Test 2: test setting parameters
    set_params!(model, [1, 2, 3])
    @test get_params(model) == [1, 2, 3]
    @test_throws ErrorException set_params!(model, [1, 2, 3, 4])
end
@testset "TB Model update_tb with several structures" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "NNaxes", "As", true)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", string(@__DIR__)*"/test_files/rllm_true.dat")
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "rcut", 5)
    strc = Structure(conf, poscar_path=gaas_poscar)
    basis = Basis(strc, conf)

    # A list of indices (in the order of params.dat) selects these parameters, it is not reduced to all or nothing
    set_value!(conf, "update_tb", "Optimizer", "1 3")
    model = TBModel([strc], [basis], comm, conf, write_summary=false)
    @test model.update == [i ∈ (1, 3) for i in eachindex(model.params)]
    reg = Hamster.Regularization(0., 0., 2)
    dL_dHr = [[ones(size(model.hs[1][1, R])) for R in axes(model.hs[1], 2)]]
    dV = Hamster.get_model_gradient(model, [1], reg, dL_dHr)
    @test all(dV[.!model.update] .== 0)
    @test any(dV[model.update] .≠ 0)

    set_value!(conf, "update_tb", "Optimizer", false)
    @test !any(TBModel([strc], [basis], comm, conf, write_summary=false).update)
    set_value!(conf, "update_tb", "Optimizer", true)
    @test all(TBModel([strc], [basis], comm, conf, write_summary=false).update)
    @test_throws ErrorException TBModel([strc], [basis], comm, conf, update_tb=[true, false], write_summary=false)

    # Indices beyond the number of parameters are an error
    set_value!(conf, "update_tb", "Optimizer", "1 1000")
    @test_throws ErrorException TBModel([strc], [basis], comm, conf, write_summary=false)
end

@testset "Test init_params keywords and files" begin
    @test Hamster.get_init_mode("ones") == :ones
    @test Hamster.get_init_mode("Zeros") == :zeros
    @test Hamster.get_init_mode("rand") == :rand
    @test Hamster.get_init_mode("random") == :rand
    @test Hamster.get_init_mode("z") == :zeros
    @test_throws ErrorException Hamster.get_init_mode("no_such_file.dat")

    # A parameter file whose name starts with o, r or z is read as a file
    labels = Hamster.string_to_param_label.(["NN1_Ga+Ga_ssσ", "NN1_As+Ga_ssσ", "NN1_As+As_ssσ"])
    mktempdir() do dir
        for name in ["old_params.dat", "results_params.dat", "zahn_params.dat"]
            file = joinpath(dir, name)
            write(file, "NN1_Ga+Ga_ssσ  2.5\nNN1_As+Ga_ssσ  -1.5\n")
            @test Hamster.get_init_mode(file) == :file
            model = TBModel(nothing, ones(3), labels, [collect(1:3)], fill(true, 3))   # as in the TBModel constructors
            # The parameter that is missing in the file keeps the value 1 and is reported
            @test_logs (:warn, r"1 of 3 TB parameters are not in") Hamster.init_params!(model, initas=file)
            @test model.params == [2.5, -1.5, 1.0]
            @test_logs Hamster.init_params!(model, initas=file, warn=false)
        end
    end
end
