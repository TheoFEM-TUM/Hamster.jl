function run_optimization(conf::Config)
    train_config_inds, val_conf_inds = get_config_index_sample(conf)

    Rs = get_translation_vectors(conf)
    
    train_strcs = get_structures(conf, config_indices=train_config_inds, Rs=Rs)
    train_bases = Basis[Basis(strc, conf) for strc in train_strcs]
    ham_train = EffectiveHamiltonian(train_strcs, train_bases, conf)

    val_strcs = get_structures(conf, config_indices=val_conf_inds, Rs=Rs)
    val_bases = Basis[Basis(strc, conf) for strc in val_strcs]
    ham_val = validate ? EffectiveHamiltonian(val_strcs, val_bases, conf) : get_empty_effective_hamiltonian()

    dl = DataLoader(train_config_inds, val_conf_inds, 8, 8, conf)
    Nε, Nk = get_neig_and_nk(dl.train_data)
    optim = GDOptimizer(Nε, Nk, conf)
    prof = HamsterProfiler(3, conf)
    
    optimize_model!(ham_train, ham_val, optim, dl, prof, conf)
end