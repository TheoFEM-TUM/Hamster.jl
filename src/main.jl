function main(comm, conf; rank=0, nranks=1, num_nodes=1, verbosity=get_verbosity(conf))
    # Clear output files from previous Hamster runs
    clear_hamster_output()

    # Create a temporary folder
    if "tmp" in readdir(pwd()) && rank == 0; rm("tmp", recursive=true); end
    if !("tmp" in readdir(pwd())) && rank == 0; mkdir("tmp"); end
    
    set_seed!(conf, rank=rank)
    set_heap_size_hint!(comm, rank=rank, verbosity=verbosity)
    hostname = readchomp(`hostname`)
    hostnames = MPI.gather(hostname, comm, root=0)
    num_gpus = get_num_gpus(conf, comm, hostname)
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
        write_block_summary("Parallelization", num_nodes=num_nodes, num_gpus=num_gpus, nhamster=nranks, 
            nstrc_per_hamster=round(Int64, Nconf/nranks), nstrc_per_node=round(Int64, Nconf/num_nodes), 
            julia_num_threads=julia_num_threads, nthreads_kpoints=nthreads_kpoints, 
            nthreads_bands=nthreads_bands, nthreads_blas=nthreads_blas)
    end
    task = decide_which_task_to_perform(conf)
    out = run_calculation(task, comm, conf, rank=rank, nranks=nranks)
    save(out, rank, filename="hamster_out.h5")
    if rank == 0
        rm("tmp", recursive=true, force=true)
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

"""
    cgroup_memory_limit() -> Union{UInt64, Nothing}

Returns the smallest memory limit (`memory.max`, cgroup v2) of the cgroup of this process and all its parent cgroups, or
`nothing` if there is none (or it cannot be read). Slurm sets the memory limit of a job (`--mem`) on the cgroup of the job,
a parent of the cgroup of the job step, which Julia itself does not take into account.
"""
function cgroup_memory_limit()
    limit = nothing
    try
        isfile("/proc/self/cgroup") || return nothing
        for line in eachline("/proc/self/cgroup")
            startswith(line, "0::") || continue
            path = String(line[4:end])
            while true
                file = joinpath("/sys/fs/cgroup", lstrip(path, '/'), "memory.max")
                if isfile(file)
                    value = strip(read(file, String))
                    if value != "max"
                        bytes = parse(UInt64, value)
                        limit = limit === nothing ? bytes : min(limit, bytes)
                    end
                end
                (path == "/" || isempty(path)) && break
                path = dirname(path)
            end
        end
    catch
        return limit
    end
    return limit
end

"""
    set_heap_size_hint!(comm; rank=0, verbosity=1)

Sets the heap size hint of the garbage collector (as `julia --heap-size-hint`) to the share of this rank of the memory
of its node when several ranks share a node or the memory is limited (e.g. by the `--mem` of a Slurm job).
Without it, every rank assumes that it can use the whole memory of the node and only collects its garbage late,
such that the ranks of a node together run out of memory although their live data would fit.

The memory of the node is the smallest of the physical memory, the cgroup limit (see `cgroup_memory_limit`) and
`SLURM_MEM_PER_NODE`. Every rank gets `0.8 × memory / nranks_on_node - 1 GiB` (but at least half of its share), where
1 GiB accounts for the memory outside of the heap (Julia runtime, compiled code, MPI buffers).
Nothing is changed if the hint was given at startup (`julia --heap-size-hint` or the environment variable
`JULIA_HEAP_SIZE_HINT`).

Setting the limit at runtime (`jl_gc_set_max_memory`) makes the garbage collector collect earlier, but it is weaker than
setting it at startup: Julia also adapts its collection heuristics to a hint given at startup. Prefer a hint at startup for
large runs, e.g. in the Slurm job script (with a margin, since a hint below the live data makes every allocation trigger
a collection, which slows the run down drastically):
    export JULIA_HEAP_SIZE_HINT=\$(( SLURM_MEM_PER_NODE * 7 / 10 / SLURM_NTASKS_PER_NODE ))M
"""
function set_heap_size_hint!(comm; rank=0, verbosity=1)
    (Base.JLOptions().heap_size_hint > 0 || !isempty(get(ENV, "JULIA_HEAP_SIZE_HINT", ""))) && return nothing
    node_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    nranks_node = MPI.Comm_size(node_comm)
    MPI.free(node_comm)

    physical = UInt64(Sys.total_physical_memory())
    memory = min(physical, UInt64(Sys.total_memory()))
    cgroup_limit = cgroup_memory_limit()
    cgroup_limit !== nothing && (memory = min(memory, cgroup_limit))
    slurm_mem = tryparse(Int64, get(ENV, "SLURM_MEM_PER_NODE", ""))
    slurm_mem !== nothing && slurm_mem > 0 && (memory = min(memory, UInt64(slurm_mem) * 2^20))
    # A single unconstrained process keeps the default of Julia
    (nranks_node == 1 && memory == physical) && return nothing

    share = memory / nranks_node
    hint = max(0.8 * share - 2^30, 0.5 * share)
    ccall(:jl_gc_set_max_memory, Cvoid, (UInt64,), floor(UInt64, hint))
    if rank == 0 && verbosity > 0
        @info @sprintf("Heap size hint of the garbage collector: %.2f GiB per rank (%d ranks share %.1f GiB on node of rank 0)",
                       hint / 2^30, nranks_node, memory / 2^30)
    end
    return nothing
end
