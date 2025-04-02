@testset "GaAs SOC" begin
    path = joinpath(@__DIR__, "test_files/")
    conf = get_config(filename=joinpath(path, "hconf_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_gaas.dat"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "verbosity", 0)

    strc = Structure(conf); basis = Basis(strc, conf)
    eff_ham = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    ks = read_from_file(joinpath(path, "kpoints.dat"))
    Hk = get_hamiltonian(eff_ham, 1, ks)
    Es_nsoc, vs = diagonalize(Hk)

    @test length(eff_ham.models) == 2
    @test size(Es_nsoc, 1) == 16

    _, Es_dft, _ = Hamster.read_eigenval(joinpath(path, "EIGENVAL_soc"), 16)
    L_nsoc = mean(abs.(Es_nsoc .- Es_dft))

    set_value!(conf, "init_params", "SOC", joinpath(path, "params.dat"))
    eff_ham_soc = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    ks = read_from_file(joinpath(path, "kpoints.dat"))
    Hk_soc = get_hamiltonian(eff_ham_soc, 1, ks)
    Es_soc, vs = diagonalize(Hk_soc)
    L_soc = mean(abs.(Es_soc .- Es_dft))
    @test L_soc < L_nsoc
end