# Building blocks of the calculations: distribution of the structures over the MPI ranks and construction of the
# effective Hamiltonian of the local structures

"""
    local_structure_indices(conf, comm; optimize, rank=0, nranks=1, write_output=true, config_inds=nothing)

Sample the configuration indices of all systems (training and validation indices if `optimize = true`) and split them
over the MPI ranks. Returns the local training and validation indices (per system). `config_inds` (training and
validation indices per system, as returned by `get_config_inds_for_systems`) are used instead of a new sample.

The training structures are split over all ranks; with fewer training structures than ranks, the first ranks get one
each and the others are idle. The validation structures are split over the ranks with training structures, so a rank
may have training structures but no validation structure. In a validated fit, an empty validation set is an error.
"""
function local_structure_indices(conf, comm; optimize, rank=0, nranks=1, write_output=true, config_inds=nothing)
    train_inds, val_inds = isnothing(config_inds) ?
        get_config_inds_for_systems(get_systems(conf), comm, conf, rank=rank, write_output=write_output, optimize=optimize) :
        config_inds
    if optimize && get_validate(conf) && sum(length, values(val_inds), init=0) == 0
        error("The fit is validated (val_data, val_mode or val_ratio), but the validation set is empty: " *
              "val_mode = $(get_val_mode(conf)), Nconf = $(get_Nconf(conf)), val_ratio = $(get_val_ratio(conf)), " *
              "configurations $(get_Nconf_min(conf)) to $(get_Nconf_max(conf)). The validation set has " *
              "round(Nconf × val_ratio) structures that are not in the training set; increase val_ratio or Nconf_max, " *
              "or give val_config_inds.")
    end
    nactive = min(nranks, sum(length, values(train_inds), init=0))
    local_val_inds = rank < nactive ? split_indices_into_chunks(val_inds, nactive, rank=rank) : empty(val_inds)
    return split_indices_into_chunks(train_inds, nranks, rank=rank), local_val_inds
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
    build_hamiltonian(conf, comm, local_inds; mode, Rs=zeros(3, 1), rank=0, nranks=1, verbosity=get_verbosity(conf),
                      xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf), kwargs...) -> (ham, strcs, bases)

Load the local structures (`local_inds`: configuration indices per system, of the trajectory `xdatcar` with the
reference structure `sc_poscar`), build their bases and the `EffectiveHamiltonian`; further keyword arguments are
passed to `EffectiveHamiltonian`.
"""
function build_hamiltonian(conf, comm, local_inds; mode, Rs=zeros(3, 1), rank=0, nranks=1,
                           verbosity=get_verbosity(conf), xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf),
                           kwargs...)
    if rank == 0 && verbosity > 1; println("Getting structures..."); end
    begin_time = MPI.Wtime()
    strcs = mapreduce(vcat, local_inds, init=Structure[]) do (system, inds)
        get_structures(conf, config_indices=inds, Rs=Rs, mode=mode, system=system, xdatcar=xdatcar, sc_poscar=sc_poscar)
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
