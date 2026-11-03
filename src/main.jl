"""
    main(comm, conf; rank=0, nranks=1, num_nodes=1, verbosity=get_verbosity(conf))

Run the calculation of the config `conf` (standard run, optimization or hyperparameter optimization, see
[`decide_which_task_to_perform`](@ref Hamster.decide_which_task_to_perform)) on all ranks of `comm` and return its
profiler. The temporary files go to a new folder `hamster_tmp_<random>` in the run folder (see
[`TEMP_DIR`](@ref Hamster.TEMP_DIR)), which is removed at the end; with one rank also if the run fails.
"""
function main(comm, conf; rank=0, nranks=1, num_nodes=1, verbosity=get_verbosity(conf))
    # Clear output files from previous Hamster runs
    clear_hamster_output()

    temp_dir = rank == 0 ? basename(mktempdir(pwd(), prefix="hamster_tmp_", cleanup=false)) : ""
    temp_dir = MPI.bcast(temp_dir, 0, comm)
    TEMP_DIR[] = temp_dir
    out = try
        run_main(comm, conf, rank=rank, nranks=nranks, num_nodes=num_nodes, verbosity=verbosity)
    catch
        # No barrier: other ranks may be stuck in another collective call
        MPI.Comm_size(comm) == 1 && rm(temp_dir, recursive=true, force=true)
        TEMP_DIR[] = "tmp"
        rethrow()
    end
    MPI.Barrier(comm)
    rank == 0 && rm(temp_dir, recursive=true, force=true)
    TEMP_DIR[] = "tmp"
    return out
end

"""
    abort_on_error(f, comm)

Run `f()`. If it throws an error and `comm` has several ranks, print the error and stack trace of this rank, add the
error to `hamster.out` and stop all ranks with `MPI.Abort` (exit code 1). Otherwise the other ranks would wait in
their next collective call until the job's time limit. With one rank the error is rethrown. Used by the `hamster`
command; `main` itself throws errors normally.
"""
function abort_on_error(f, comm)
    try
        return f()
    catch err
        MPI.Comm_size(comm) == 1 && rethrow()
        rank = MPI.Comm_rank(comm)
        message = sprint(showerror, err, catch_backtrace())
        println(stderr, "\nERROR on MPI rank $rank (aborting all ranks):\n", message)
        flush(stderr)
        try
            open(file -> println(file, "\nERROR on MPI rank $rank: ", first(split(sprint(showerror, err), '\n'))),
                 "hamster.out", "a")
        catch
        end
        MPI.Abort(comm, 1)
    end
end

"""
    run_main(comm, conf; rank=0, nranks=1, num_nodes=1, verbosity=get_verbosity(conf))

The part of [`main`](@ref Hamster.main) that runs in the temporary folder of the run: output header, the calculation
and the combination of the per-structure eigenvalue files.
"""
function run_main(comm, conf; rank=0, nranks=1, num_nodes=1, verbosity=get_verbosity(conf))
    set_seed!(conf, rank=rank)
    hostnames = MPI.gather(readchomp(`hostname`), comm, root=0)
    # On every rank: otherwise the other ranks keep Julia's default (half of the Julia threads) and oversubscribe
    nthreads_blas = get_nthreads_blas(conf); BLAS.set_num_threads(nthreads_blas)
    if rank == 0
        generate_output(conf, hostnames=hostnames)
        julia_num_threads = Threads.nthreads()
        nthreads_kpoints = get_nthreads_kpoints(conf)
        nthreads_bands = get_nthreads_bands(conf)
        
        # COV_EXCL_START
        if get(ENV, "OMP_NUM_THREADS", "not set") ∉ ["1", "not set"] && rank == 0
            @warn "OMP_NUM_THREADS is not set to 1 (currently: $(get(ENV, "OMP_NUM_THREADS", "not set"))). This may hurt performance in multi-threaded or distributed settings. Consider setting OMP_NUM_THREADS=1, but test for your case."
        end
        # COV_EXCL_STOP

        Nconf = get_Nconf(conf)
        write_block_summary("Parallelization", num_nodes=num_nodes, nhamster=nranks, 
            nstrc_per_hamster=round(Int64, Nconf/nranks), nstrc_per_node=round(Int64, Nconf/num_nodes), 
            julia_num_threads=julia_num_threads, nthreads_kpoints=nthreads_kpoints, 
            nthreads_bands=nthreads_bands, nthreads_blas=nthreads_blas)
    
        blocks_to_write = string.(vcat(["Options"], keys(conf.blocks)...))
        filter!(block -> block ∈ unique([tag.block for tag in CONFIG_TAGS]), blocks_to_write)
        write_config_tags(conf; blocks=blocks_to_write, show_desc=true)
    end

    task = decide_which_task_to_perform(conf)
    out = run_calculation(task, comm, conf, rank=rank, nranks=nranks)
    save(out, rank, filename="hamster_out.h5")
    if rank == 0
        collapse_time = @elapsed collapse_eigenvalue_files(conf)
        # COV_EXCL_START
        if verbosity > 1
            println(" Final write time: $collapse_time s")
        end
        # COV_EXCL_STOP
    end
    return out
end

"""
    collapse_eigenvalue_files(conf)

Combine the eigenvalue (and eigenvector, if `save_vecs`) files of the single structures in the temporary folder into
`Es.dat` (`vs.dat`). With several systems (`train_mode = universal`), every system gets its own file `Es_<system>.dat`
(`vs_<system>.dat`), because the systems have different numbers of bands.
"""
function collapse_eigenvalue_files(conf)
    systems = get_systems(conf)
    files = get_save_vecs(conf) ? ["Es", "vs"] : ["Es"]
    for str in files, system in systems
        outname = length(systems) > 1 ? "$(str)_$system" : str
        collapse_files_with(str, location=temp_path(system), outname=outname)
    end
end

"""
    decide_which_task_to_perform(conf::Config)

Given a `Config` instance, decides which type of calculation is to be performed based on certain tags in the input file.

# Details
- An `Optimizer` without a `HyperOpt` block tells Hamster to run a parameter optimization.
- An `Optimizer` with a `HyperOpt` block tells Hamster to run a hyperparameter optimization.
- Otherwise, a standard calculation is performed.
"""
function decide_which_task_to_perform(conf::Config)
    if haskey(conf, "Optimizer") && !haskey(conf, "HyperOpt")
        return Val{:optimization}()
    elseif haskey(conf, "Optimizer") && haskey(conf, "HyperOpt")
        return Val{:hyper_optimization}()
    else
        return Val{:standard}()
    end
    error("Your given configuration does not specify which calculation to perform. Check your input!")
end