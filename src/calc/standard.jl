function run_calculation(::Val{:standard}, comm, conf::Config; rank=0, nranks=1)
    config_inds, _ = get_config_index_sample(conf)
    if get_config_inds(conf) ≠ "none"
        config_inds = read_from_file(get_config_inds(conf))
    end
   
    if rank == 0
       write_to_file(config_inds, "config_inds")
    end
    
    MPI.Bcast!(config_inds, comm, root=0)
    MPI.Barrier(comm)
 
    mode = haskey(conf, "Supercell") ? "md" : "pc"
    local_inds = split_indices_into_chunks(config_inds, nranks, rank=rank)
    strcs = get_structures(conf, config_indices=local_inds, mode=mode)
    bases = Basis[Basis(strc, conf) for strc in strcs]
    ham = EffectiveHamiltonian(strcs, bases, comm, conf, rank=rank, nranks=nranks)

    prof = HamsterProfiler(2, conf, Niter=length(local_inds), Nbatch=1)

    get_eigenvalues(ham, prof, local_inds, comm, conf, rank=rank, nranks=nranks)
    return prof
end

function get_eigenvalues(ham::EffectiveHamiltonian, prof, local_inds, comm, conf=get_empty_config(); Nbatch=get_nbatch(conf), rank=0, nranks=1, verbosity=get_verbosity(conf))
    strc_ind = 0
    ks = get_kpoints_from_config(conf)
    Nstrc_tot = MPI.Reduce(ham.Nstrc, +, comm, root=0)
    for (batch_id, indices) in enumerate(chunks(1:ham.Nstrc, n=Nbatch))
        for index in indices
            strc_ind += 1
            ham_time_local = @elapsed Hk = get_hamiltonian(ham, index, ks)
            Neig = ham.sp_diag isa Sparse ? get_neig(conf) : size(Hk[1], 1)
            diag_time_local = @elapsed Es, vs = diagonalize(Hk, Neig=Neig, target=get_eig_target(conf))

            ham_time = MPI.Reduce(ham_time_local, +, comm, root=0)
            diag_time = MPI.Reduce(diag_time_local, +, comm, root=0)

            if rank == 0
                prof.timings[1, strc_ind, 1] = ham_time / nranks
                prof.timings[1, strc_ind, 2] = diag_time / nranks
                if verbosity > 1
                    println(" Hamiltonian time: $(ham_time ./ nranks) s")
                    println(" Diagonalization time: $(diag_time ./ nranks) s")
                end
            end
            if Nstrc_tot == 1 && rank == 0
                write_to_file(Es, "Es")
                write_to_file(vs, "vs")
            else
                if !("tmp" in readdir(pwd())); mkdir("tmp"); end
                write_to_file(Es, "tmp/Es$(local_inds[index])")
                write_to_file(vs, "tmp/vs$(local_inds[index])")
            end            
            print_train_status(prof, strc_ind, batch_id, verbosity=verbosity)
        end
    end
end

function get_kpoints_from_config(conf::Config; kpoints_file=get_kpoints_file(conf))::Matrix{Float64}
    if occursin("EIGENVAL", kpoints_file)
        ks, _, _ = read_eigenval(kpoints_file)
        return ks
    elseif occursin(".h5", kpoints_file)
        ks = h5read(kpoints_file, "kpoints")
        if ks isa Matrix{Float64}
            return ks
        elseif ks isa Array{Float64, 3}
            return ks[:, :, 1]
        end
    end
end