"""
    run_calculation(::Val{:standard}, comm, conf::Config; rank=0, nranks=1)

Performs a standard calculation for an effective Hamiltonian model, computing either the eigenvalues or the Hamiltonian itself for a given set of structures.

# Workflow

1. **Configuration Sampling**  
   - Retrieve configuration indices for the selected systems using `get_config_inds_for_systems`.  
   - Write the indices to a file on the root process (`rank == 0`) and broadcast them to all ranks for consistency.  
   - Split indices among MPI ranks for parallel execution.

2. **Determine Calculation Mode**  
   - `"md"` (molecular dynamics) if the configuration contains a `Supercell` and only one system.  
   - `"universal"` if multiple systems are present with a `Supercell`.  
   - `"pc"` (primitive cell) otherwise.

3. **Structure Extraction**  
   - Load the atomic structures for the assigned configuration indices using `get_structures`.

4. **Basis Construction**  
   - Build basis functions (`Basis`) for each structure to represent the effective Hamiltonian.

5. **Hamiltonian Initialization**  
   - Initialize the `EffectiveHamiltonian` model with the structures and their bases, ready for eigenvalue or Hamiltonian computations.

6. **Profiler Setup**  
   - Create a `HamsterProfiler` to record computational timings, performance metrics, and other profiling data.

7. **Eigenvalue / Hamiltonian Calculation**  
   - Execute `get_eigenvalues` (or the equivalent routine) to compute the eigenvalues or solve the Hamiltonian for the provided structures.

# Required Inputs

- **Structural configurations**  
  - `train_mode = md`: see `get_xdatcar` and `get_sc_poscar`
  - `train_mode = pc`: see `get_poscar`
- **Eigenvalue or Hamiltonian data** as specified in the configuration.

# Output Files

- `ham.h5` — HDF5 file containing the Hamiltonian, generated if `write_hk` (k-space) or `write_hr` (real space) is `true`.
- `hamster.out` — Standard Hamster output file with summary information.  
- `hamster_out.h5` — HDF5 file containing:
  - Loss values (for iterative calculations, if applicable)  
  - Timings and profiling data
- `Es.dat` — Plain text file containing computed eigenvalues for each structure (written only if `skip_diag` is `true`; default).
"""
function run_calculation(::Val{:standard}, comm, conf::Config; rank=0, nranks=1, verbosity=get_verbosity(conf), write_output=true)
    
        systems = get_systems(conf)
    config_inds, _ = get_config_inds_for_systems(systems, comm, conf, rank=rank, write_output=write_output, optimize=false)
    local_inds = split_indices_into_chunks(config_inds, nranks, rank=rank)

    has_data = !isempty(local_inds)
    color = has_data ? 1 : nothing
    comm_active = MPI.Comm_split(comm, color, rank)

    all_has_data = MPI.gather(Int(has_data), comm, root=0)
    if rank == 0
        for (r, flag) in enumerate(all_has_data)
            if flag == 0
                append_output_line("Warning: Rank $(r-1) has no data and will be idle. Revise your parallelization settings!")
            end
        end
    end

    if has_data
        active_rank = MPI.Comm_rank(comm_active)
        active_size = MPI.Comm_size(comm_active)
        mode = haskey(conf, "Supercell") ? (length(systems) > 1 ? "universal" : "md") : "pc"
        if rank == 0 && verbosity > 1; println("Getting structures..."); end
        begin_time = MPI.Wtime()
        strcs = mapreduce(vcat, local_inds, init=Structure[]) do (system, inds)
            get_structures(conf, config_indices=inds, mode=mode, system=system)
        end
        strc_time = MPI.Wtime() - begin_time
        if rank == 0 && verbosity > 1; println(" Structure time: $strc_time s"); end

        if rank == 0 && verbosity > 1; println("Getting bases..."); end
        begin_time = MPI.Wtime()
        bases = Basis[Basis(strc, conf, comm=comm_active) for strc in strcs]
        bases_time = MPI.Wtime() - begin_time
        if rank == 0 && verbosity > 1; println(" Basis time: $bases_time s"); end

        if rank == 0 && verbosity > 1; println("Getting Hamiltonian models..."); end
        begin_time = MPI.Wtime()
        ham = EffectiveHamiltonian(strcs, bases, comm_active, conf, rank=active_rank, nranks=active_size)
        ham_time = MPI.Wtime() - begin_time
        if rank == 0 && verbosity > 1; println(" Model time: $ham_time s"); end

        Niter = sum(length.(values(local_inds)))
        prof = HamsterProfiler(3, conf, Niter=Niter, Nbatch=1)

        get_eigenvalues(ham, prof, local_inds, comm_active, conf, rank=active_rank, nranks=active_size)

        if rank == 0 && verbosity > 1; println("Post processing..."); end
        begin_time = MPI.Wtime()
        run_post_processing(strcs, bases, local_inds, comm_active, conf, rank=active_rank, nranks=active_size)
        post_time = MPI.Wtime() - begin_time
        if rank == 0 && verbosity > 1; println(" Post-processing time: $post_time s"); end
    else
        prof = HamsterProfiler(1, conf)
    end
    MPI.Barrier(comm)

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
        verbosity=get_verbosity(conf))
    
    strc_ind = 0
    ks = get_kpoints_from_config(conf)
    Nstrc_tot = MPI.Reduce(ham.Nstrc, +, comm, root=0)

    for (batch_id, indices) in enumerate(chunks(1:ham.Nstrc, n=Nbatch))
        for index in indices
            strc_ind += 1
            system, config_index = get_system_and_config_index(index, local_inds)
            ham_time_local = @elapsed Hk = get_hamiltonian(ham, index, ks, comm, write_hr=write_hr, config_index=config_index, system=system, rank=rank, nranks=nranks, ham_file=ham_file)
            Neig = ham.sp_diag isa Sparse ? get_neig(conf) : size(Hk[1], 1)

            Es = zeros(1, 1); vs = zeros(ComplexF64, 1, 1, 1)
            diag_time_local = @elapsed begin
                if !skip_diag
                    Es, vs = diagonalize(Hk, Neig=Neig, target=get_eig_target(conf), method=get_diag_method(conf))
                end
            end

            ham_time = MPI.Reduce(ham_time_local, +, comm, root=0)
            diag_time = MPI.Reduce(diag_time_local, +, comm, root=0)

            write_begin = MPI.Wtime()
            if write_hk
                write_ham(Hk, ks, comm, config_index, filename=ham_file, system=system, rank=rank, nranks=nranks)
            end
            if Nstrc_tot == 1 && rank == 0
                if !skip_diag; write_to_file(Es, "Es"); end
                if save_vecs && !skip_diag; write_to_file(vs, "vs"); end
            else
                if !("tmp" in readdir(pwd())) && rank == 0; mkdir("tmp"); end
                MPI.Barrier(comm)
                if !skip_diag; write_to_file(Es, "tmp/Es$config_index"); end
                if save_vecs && !skip_diag; write_to_file(vs, "tmp/vs$config_index"); end
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

function run_post_processing(strcs, bases, local_inds, comm, conf=get_empty_config(); rank=0, nranks=1, 
            write_current_file=get_write_current(conf), current_file=get_current_file(conf), ham_file=get_ham_file(conf))

    if write_current_file
        for index in eachindex(strcs)
            system, config_index = get_system_and_config_index(index, local_inds)
            bonds = get_bonds(strcs[index], bases[index], conf)
            write_current(bonds, comm, config_index; ham_file=ham_file, filename=current_file, system=system, rank=rank, nranks=nranks)
        end
    end
end