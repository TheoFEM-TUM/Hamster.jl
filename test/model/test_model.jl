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