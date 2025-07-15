function main(comm, conf; rank=0, nranks=1, num_nodes=1, verbosity=get_verbosity(conf))
    set_seed!(conf, rank=rank)
    hostnames = MPI.gather(readchomp(`hostname`), comm, root=0)
    if rank == 0
        generate_output(conf, hostnames=hostnames)
        julia_num_threads = Threads.nthreads()
        nthreads_kpoints = get_nthreads_kpoints(conf)
        nthreads_bands = get_nthreads_bands(conf)
        nthreads_blas = get_nthreads_blas(conf); BLAS.set_num_threads(nthreads_blas)
        
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
    end
    task = decide_which_task_to_perform(conf)
    out = run_calculation(task, comm, conf, rank=rank, nranks=nranks)
    if isdir("tmp") && rank == 0
        files = ["Es"]
        if get_save_vecs(conf); push!(files, "vs"); end
        collapse_time = @elapsed for str in files
            if any([occursin(str, file) for file in readdir(joinpath(pwd(), "tmp"))])
                collapse_files_with(str)
            end
        end
        rm("tmp", recursive=true)
        # COV_EXCL_START
        if verbosity > 1
            println(" Final write time: $collapse_time s")
        end
        # COV_EXCL_STOP
    end
    return out
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