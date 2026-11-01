# Building blocks of the calculations: distribution of the structures over the MPI ranks and construction of the
# effective Hamiltonian of the local structures

"""
    local_structure_indices(conf, comm; optimize, rank=0, nranks=1, write_output=true)

Sample the configuration indices of all systems (training and validation indices if `optimize = true`) and split them
over the MPI ranks. Returns the local training and validation indices (per system).
"""
function local_structure_indices(conf, comm; optimize, rank=0, nranks=1, write_output=true)
    train_inds, val_inds = get_config_inds_for_systems(get_systems(conf), comm, conf, rank=rank, write_output=write_output,
                                                       optimize=optimize)
    return split_indices_into_chunks(train_inds, nranks, rank=rank), split_indices_into_chunks(val_inds, nranks, rank=rank)
end

"""
    active_communicator(has_data, comm; rank=0)

Communicator of the ranks that have data. A warning is written to `hamster.out` for every idle rank.
"""
function active_communicator(has_data, comm; rank=0)
    comm_active = MPI.Comm_split(comm, has_data ? 1 : nothing, rank)
    all_has_data = MPI.gather(Int(has_data), comm, root=0)
    if rank == 0
        for (r, flag) in enumerate(all_has_data)
            if flag == 0
                append_output_line("Warning: Rank $(r-1) has no data and will be idle. Revise your parallelization settings!")
            end
        end
    end
    return comm_active
end

"""
    build_hamiltonian(conf, comm, local_inds; mode, Rs=zeros(3, 1), reindex=false, rank=0, nranks=1,
                      verbosity=get_verbosity(conf), kwargs...) -> (ham, strcs, bases)

Load the local structures (`local_inds`: configuration indices per system), build their bases and the
`EffectiveHamiltonian`; further keyword arguments are passed to `EffectiveHamiltonian`. With `reindex = true`,
`local_inds` of a system with fewer structures than requested is reset to `1:n`.
"""
function build_hamiltonian(conf, comm, local_inds; mode, Rs=zeros(3, 1), reindex=false, rank=0, nranks=1,
                           verbosity=get_verbosity(conf), kwargs...)
    if rank == 0 && verbosity > 1; println("Getting structures..."); end
    begin_time = MPI.Wtime()
    strcs = mapreduce(vcat, local_inds, init=Structure[]) do (system, inds)
        system_strcs = get_structures(conf, config_indices=inds, Rs=Rs, mode=mode, system=system)
        if reindex && length(system_strcs) < length(inds)
            local_inds[system] = collect(1:length(system_strcs))
        end
        return system_strcs
    end
    strc_time = MPI.Wtime() - begin_time
    if rank == 0 && verbosity > 1; println(" Structure time: $strc_time s"); end

    if rank == 0 && verbosity > 1; println("Getting bases..."); end
    begin_time = MPI.Wtime()
    bases = Basis[Basis(strc, conf, comm=comm) for strc in strcs]
    bases_time = MPI.Wtime() - begin_time
    if rank == 0 && verbosity > 1; println(" Basis time: $bases_time s"); end

    if rank == 0 && verbosity > 1; println("Getting Hamiltonian models..."); end
    begin_time = MPI.Wtime()
    ham = EffectiveHamiltonian(strcs, bases, comm, conf; rank=rank, nranks=nranks, kwargs...)
    ham_time = MPI.Wtime() - begin_time
    if rank == 0 && verbosity > 1; println(" Model time: $ham_time s"); end
    return ham, strcs, bases
end
