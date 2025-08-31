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
function run_calculation(::Val{:standard}, comm, conf::Config; rank=0, nranks=1, verbosity=get_verbosity(conf))
    config_inds, _ = get_config_index_sample(conf)
   
    if rank == 0
       write_to_file(config_inds, "config_inds")
    end
    
    MPI.Bcast!(config_inds, comm, root=0)
    MPI.Barrier(comm)
 
    mode = haskey(conf, "Supercell") ? "md" : "pc"
    local_inds = split_indices_into_chunks(config_inds, nranks, rank=rank)
    
    if rank == 0 && verbosity > 1; println("Getting structures..."); end
    begin_time = MPI.Wtime()
    strcs = get_structures(conf, config_indices=local_inds, mode=mode)
    strc_time = MPI.Wtime() - begin_time
    if rank == 0 && verbosity > 1; println(" Structure time: $strc_time s"); end

    if rank == 0 && verbosity > 1; println("Getting bases..."); end
    begin_time = MPI.Wtime()
    bases = Basis[Basis(strc, conf) for strc in strcs]
    bases_time = MPI.Wtime() - begin_time
    if rank == 0 && verbosity > 1; println(" Basis time: $bases_time s"); end

    if rank == 0 && verbosity > 1; println("Getting Hamiltonian models..."); end
    begin_time = MPI.Wtime()
    ham = EffectiveHamiltonian(strcs, bases, comm, conf, rank=rank, nranks=nranks)
    ham_time = MPI.Wtime() - begin_time
    if rank == 0 && verbosity > 1; println(" Model time: $ham_time s"); end

    prof = HamsterProfiler(3, conf, Niter=length(local_inds), Nbatch=1)

    get_eigenvalues(ham, prof, local_inds, comm, conf, rank=rank, nranks=nranks)
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
function get_eigenvalues(ham::EffectiveHamiltonian, prof, local_inds, comm, conf=get_empty_config(); Nbatch=get_nbatch(conf), save_vecs=get_save_vecs(conf), rank=0, nranks=1, verbosity=get_verbosity(conf))
    strc_ind = 0
    ks = get_kpoints_from_config(conf)
    Nstrc_tot = MPI.Reduce(ham.Nstrc, +, comm, root=0)

    for (batch_id, indices) in enumerate(chunks(1:ham.Nstrc, n=Nbatch))
        for index in indices
            strc_ind += 1
            ham_time_local = @elapsed Hk = get_hamiltonian(ham, index, ks)
            Neig = ham.sp_diag isa Sparse ? get_neig(conf) : size(Hk[1], 1)
            diag_time_local = @elapsed Es, vs = diagonalize(Hk, Neig=Neig, target=get_eig_target(conf), method=get_diag_method(conf))

            ham_time = MPI.Reduce(ham_time_local, +, comm, root=0)
            diag_time = MPI.Reduce(diag_time_local, +, comm, root=0)

            write_begin = MPI.Wtime()
            if Nstrc_tot == 1 && rank == 0
                write_to_file(Es, "Es")
                if save_vecs; write_to_file(vs, "vs"); end
            else
                if !("tmp" in readdir(pwd())) && rank == 0; mkdir("tmp"); end
                MPI.Barrier(comm)
                write_to_file(Es, "tmp/Es$(local_inds[index])")
                if save_vecs; write_to_file(vs, "tmp/vs$(local_inds[index])"); end
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
    elseif occursin("gamma", lowercase(kpoints_file))
        return zeros(3, 1)
    end
end