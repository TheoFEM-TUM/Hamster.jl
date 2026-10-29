"""
    run_calculation(::Val{:standard}, comm, conf::Config; rank=0, nranks=1)

Performs a standard calculation for an effective Hamiltonian model.

# Arguments
- `::Val{:standard}`: A type parameter indicating that this function performs a standard calculation.
- `comm`: The MPI communicator used for parallel processing.
- `conf::Config`: A configuration object that contains parameters for the calculation.
- `rank`: The MPI rank of the current process (default: `0`).
- `nranks`: The total number of MPI ranks (default: `1`).

# Function Behavior
1. Retrieves the configuration indices using `get_config_index_sample(conf)`.
2. If configuration indices are specified in the configuration file, they are read from a file.
3. The root process (`rank == 0`) writes the configuration indices to a file.
4. The indices are broadcast to all processes using `MPI.Bcast!`, ensuring consistency across ranks.
5. Determines the mode of calculation:
   - `"md"` (molecular dynamics) if the configuration contains `"Supercell"`.
   - `"pc"` (phonon calculation) otherwise.
6. Distributes the configuration indices among MPI ranks for parallel execution.
7. Extracts atomic structures using `get_structures(conf, config_indices=local_inds, mode=mode)`.
8. Constructs bases using the `Basis` type for each structure.
9. Initializes the `EffectiveHamiltonian` for solving electronic or vibrational properties.
10. Sets up a `HamsterProfiler` to profile the computation over multiple iterations.
11. Computes eigenvalues using `get_eigenvalues`, which performs the main calculation.
12. Returns the profiler object (`prof`), which contains performance and profiling data.

# Returns
- `prof::HamsterProfiler`: An object containing profiling information about the Hamiltonian calculation.
"""
function run_calculation(::Val{:standard}, comm, conf::Config; rank=0, nranks=1, verbosity=get_verbosity(conf), write_output=true)
    systems = get_systems(conf)
    if rank == 0
      println("Rank 0 Number of Systems: ", length(systems))
      println("Rank 0 Number of ranks $nranks")
    end
    config_inds, _ = get_config_inds_for_systems(systems, comm, conf, rank=rank, write_output=write_output, optimize=false)
    local_inds = split_indices_into_chunks(config_inds, nranks, rank=rank)
    mode = haskey(conf, "Supercell") ? "universal" : "pc"
    if rank == 0 && verbosity > 1; println("Getting structures..."); end
    begin_time = MPI.Wtime()
    strcs = mapreduce(vcat, local_inds, init=Structure[]) do (system, inds)
        get_structures(conf, config_indices=inds, mode=mode, system=system)
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
    ham = EffectiveHamiltonian(strcs, bases, comm, conf, rank=rank, nranks=nranks, simmat_prefix="simmat_standard", sparse_iterators=false)
    ham_time = MPI.Wtime() - begin_time
    if rank == 0 && verbosity > 1; println(" Model time: $ham_time s"); end

    Niter = sum(length.(values(local_inds)))
    prof = HamsterProfiler(3, conf, Niter=Niter, Nbatch=1)

    get_eigenvalues(ham, prof, local_inds, comm, conf, rank=rank, nranks=nranks)

    if haskey(conf, "Unfold")
        if rank == 0 && verbosity > 0; println("Unfolding bands..."); end
        run_unfold(ham, strcs, bases, local_inds, systems, comm, conf, rank=rank, nranks=nranks)
    end
    return prof
end

"""
    get_eigenvalues(ham::EffectiveHamiltonian, prof, local_inds, comm, conf=get_empty_config();
                    Nbatch=get_nbatch(conf), rank=0, nranks=1, verbosity=get_verbosity(conf))

Computes eigenvalues and eigenvectors of the Hamiltonian for a set of structures, distributing the computation across MPI ranks.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object.
- `prof`: A profiling object that stores timing information for each step.
- `local_inds`: Indices of the local structures assigned to the current MPI rank.
- `comm`: The MPI communicator used for parallel execution.
- `conf`: Configuration object (default: `get_empty_config()`) containing parameters for diagonalization.
- `Nbatch`: The batch size for processing structures (default: `get_nbatch(conf)`).
- `rank`: The rank of the MPI process (default: `0`).
- `nranks`: Total number of MPI ranks (default: `1`).
- `verbosity`: Level of verbosity for printed output (default: `get_verbosity(conf)`).
"""
function get_eigenvalues(ham::EffectiveHamiltonian, prof, local_inds, comm, conf=get_empty_config(); 
        Nbatch=get_nbatch(conf), 
        save_vecs=get_save_vecs(conf), 
        rank=0, 
        nranks=1, 
        write_hk=get_write_hk(conf),
        write_hr=get_write_hr(conf),
        ham_file=get_ham_file(conf),
        skip_diag=get_skip_diag(conf),
        eigen_file="eval_eigenval.h5",
        verbosity=get_verbosity(conf))
    
    strc_ind = 0
    
    # A single structure of an unnamed system is written to Es.dat (and vs.dat), everything else to `eigen_file`
    Nstrc_tot = MPI.Allreduce(ham.Nstrc, +, comm)
    Nstrc_max = MPI.Allreduce(ham.Nstrc, max, comm)
    named_systems = MPI.Allreduce(Int(any(!isempty, keys(local_inds))), max, comm) == 1
    write_h5 = Nstrc_tot > 1 || named_systems
    if write_h5 && rank == 0 && isfile(eigen_file); rm(eigen_file); end
    MPI.Barrier(comm)

    for (batch_id, indices) in enumerate(chunks(1:ham.Nstrc, n=Nbatch))
        if Nbatch > 1; load_simmats!(ham, indices); end
        for index in indices
            strc_ind += 1
            system, config_index = get_system_and_config_index(index, local_inds)
            ks = get_kpoints_from_config(conf, system = system)
            ham_time_local = @elapsed Hk = get_hamiltonian(ham, index, ks, comm, write_hr=write_hr, config_index=config_index, system=system, rank=rank, nranks=nranks)
            Neig = ham.sp_diag isa Sparse ? get_neig(conf) : hamiltonian_dim(Hk)

            Es = zeros(1, 1); vs = zeros(ComplexF64, 1, 1, 1)
            diag_time_local = @elapsed begin
                if !skip_diag
                    Es, vs = diagonalize(Hk, Neig=Neig, target=get_eig_target(conf), method=get_diag_method(conf), vectors=save_vecs)
                end
            end

            ham_time = MPI.Reduce(ham_time_local, +, comm, root=0)
            diag_time = MPI.Reduce(diag_time_local, +, comm, root=0)

            write_begin = MPI.Wtime()
            if write_hk
                write_ham(host_hamiltonians(Hk), ks, comm, config_index, filename=ham_file, system=system, rank=rank, nranks=nranks)
            end
            if !skip_diag && write_h5
                for r in 0:nranks-1
                    if r == rank; append_eigen_output(Es, vs, system, config_index, filename=eigen_file, save_vecs=save_vecs); end
                    MPI.Barrier(comm)
                end
            elseif !skip_diag
                write_to_file(Es, "Es")
                if save_vecs; write_to_file(vs, "vs"); end
            end
            write_time = MPI.Wtime() - write_begin
            prof.timings[1, strc_ind, 3] = write_time
            if rank == 0
                prof.timings[1, strc_ind, 1] = ham_time / nranks
                prof.timings[1, strc_ind, 2] = diag_time / nranks
                # COV_EXCL_START
                if verbosity > 1
                    println(" Hamiltonian time: $(ham_time ./ nranks) s")
                    println(" Diagonalization time: $(diag_time ./ nranks) s")
                    println(" Write time: $write_time s")
                end
                # COV_EXCL_STOP
            end

            print_train_status(prof, strc_ind, batch_id, verbosity=verbosity)
        end
        if Nbatch > 1; unload_simmats!(ham, indices); end
    end

    # Ranks with fewer structures take part in the collective MPI calls of the remaining iterations of the other ranks
    for _ in ham.Nstrc+1:Nstrc_max
        if write_hr; wait_for_writing_ranks(comm, nranks); end
        MPI.Reduce(0.0, +, comm, root=0)
        MPI.Reduce(0.0, +, comm, root=0)
        if write_hk; wait_for_writing_ranks(comm, nranks); end
        if !skip_diag && write_h5; wait_for_writing_ranks(comm, nranks); end
    end
end

# Matches the barriers of a rank-by-rank write (e.g., `write_ham`) without writing anything
wait_for_writing_ranks(comm, nranks) = for _ in 1:nranks; MPI.Barrier(comm); end

"""
    get_kpoints_from_config(conf; kpoints_file=get_kpoints_file(conf), system="") -> Matrix{Float64}

The k-points (fractional, 3 × Nk) of `system`: from an `EIGENVAL` or a VASP `KPOINTS` file (the same k-points for all
systems), from the group `system` of an `.h5` file (dataset `kpoints`), or only the Γ point (`gamma`).
"""
function get_kpoints_from_config(conf::Config; kpoints_file=get_kpoints_file(conf), system ="")::Matrix{Float64}
    if occursin("EIGENVAL", kpoints_file)
        ks, _, _ = read_eigenval(kpoints_file)
        return ks
    elseif occursin(".h5", kpoints_file)
        ks = h5read(kpoints_file, "$system/kpoints")
        if ks isa Matrix{Float64}
            return ks
        elseif ks isa Array{Float64, 3}
            return ks[:, :, 1]
        end
    elseif occursin("KPOINTS", basename(kpoints_file))
        return read_kpoints(kpoints_file)
    elseif occursin("gamma", lowercase(kpoints_file))
        return zeros(3, 1)
    end
    error("Unknown k-points '$kpoints_file'. Valid options are gamma, an EIGENVAL or KPOINTS file and an .h5 file.")
end