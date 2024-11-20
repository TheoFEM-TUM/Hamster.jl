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
    Hr = get_hr(model, model.V)
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
    # Test parameter initialization
    model = TBModel(nothing, zeros(3), [true, true, true])
    basis = nothing
    Hamster.init_params!(model, basis, initas="ones")
    @test model.V == ones(3)
    Hamster.init_params!(model, basis, initas="random")
    @test all(0 .< model.V .< 1)
end