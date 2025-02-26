function run_calculation(::Val{:standard}, comm, conf::Config; rank=0, nranks=1)
    config_inds, _ = get_config_index_sample(conf)
   
    if rank == 0
       write_to_file(train_config_inds, "train_config_inds")
    end
    
    MPI.Bcast!(train_config_inds, comm, root=0)
    MPI.Barrier(comm)
 
    mode = haskey(conf, "Supercell") ? "md" : "pc"
    local_inds = split_indices_into_chunks(config_inds, nranks, rank=rank)
    strcs = get_structures(conf, config_indices=local_inds, Rs=Rs, mode=mode)
    bases = Basis[Basis(strc, conf) for strc in train_strcs]
    ham = EffectiveHamiltonian(strcs, bases, comm, conf, rank=rank, nranks=nranks)

    # TODO: HamsterProfiler

    get_eigenvalues(ham, conf)
end

function get_eigenvalues(ham::EffectiveHamiltonian, conf=get_empty_config(); Nbatch=get_nbatch(conf))
    for (batch_id, indices) in enumerate(chunks(1:eff_ham.Nstrc, n=Nbatch))
        for index in indices
            Hk = get_hamiltonian(eff_ham, index, ks)
            Es, vs = diagonalize(Hk)
        end
    end
end