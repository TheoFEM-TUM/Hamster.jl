begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))

    # Test that effective Hamiltonian model gives correct eigenvalues
    ham_train = EffectiveHamiltonian(conf)
    optim = GDOptimizer(conf)
    dl = DataLoader([1], [1], 8, 8, conf)
    ham_val = Hamster.get_empty_effective_hamiltonian()
    optimize_model!(ham_train, ham_val, optim, dl, conf)
end