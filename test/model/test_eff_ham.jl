path = string(@__DIR__) * "/test_files/"

@testset "EffectiveHamiltonian for PC" begin
    conf = get_empty_config()
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "NNaxes", "As", true)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", joinpath(path, "rllm_true.dat"))
    set_value!(conf, "rcut", 5)
    set_value!(conf, "rcut_tol", 100)
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "verbosity", 0)

    # Test that effective Hamiltonian model gives correct eigenvalues
    strc = Structure(conf); basis = Basis(strc, conf)
    eff_ham = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    ks = read_from_file(joinpath(path, "kpoints.dat"))
    Es_correct = read_from_file(joinpath(path, "Es.dat"))
    Hk = get_hamiltonian(eff_ham, 1, ks)
    Es, vs = diagonalize(Hk)
    @test mean(abs.(Es .- Es_correct)) < 0.01

    # Test empty model
    eff_empty = EffectiveHamiltonian([], [], comm, conf, rank=rank, nranks=nranks)
    @test eff_empty.Nstrc == 0
    @test eff_empty.models === nothing
end