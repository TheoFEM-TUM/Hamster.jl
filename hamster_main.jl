using MPI, Printf

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Intel MKL for BLAS/LAPACK if it is installed (hamster --install --mkl). MKL's threads follow the Julia threads (by
# default, MKL takes its thread count from OMP_NUM_THREADS, which is usually 1 for Hamster runs).
if !isnothing(Base.find_package("MKL"))
    haskey(ENV, "MKL_NUM_THREADS") || (ENV["MKL_NUM_THREADS"] = string(Threads.nthreads()))
    @eval using MKL
end

# Only rank 0 precompiles
rank == 0 && @eval using Hamster
MPI.Barrier(comm)
rank ≠ 0 && @eval using Hamster

num_nodes = parse(Int64, get(ENV, "SLURM_JOB_NUM_NODES", "1"))

args = Hamster.parse_commandline(ARGS)

conf = haskey(args, "conf") ? get_config(filename=args["conf"]) : get_config()
for (key, value) in args
    set_value!(conf, key, value)
end
# Tags on the command line set Options tags: warn about unknown and misplaced ones
if rank == 0
    cli_conf = Hamster.get_empty_config()
    foreach(((key, value),) -> key == "conf" || set_value!(cli_conf, key, value), args)
    foreach(message -> @warn(replace(message, "in the Options block" => "on the command line (Options block)")),
            Hamster.check_config_tags(cli_conf))
end
if rank ≠ 0; set_value!(conf, "verbosity", 0); end

# Load the GPU extensions (CUDA.jl and CUDSS.jl must be installed, e.g., in the default environment)
if lowercase(Hamster.get_device(conf)) == "gpu"
    @eval using CUDA
    try
        @eval using CUDSS
    catch
        rank == 0 && @warn "CUDSS.jl could not be loaded. The shift-invert eigensolver is not available on the GPU."
    end
end

begin_time = MPI.Wtime()
Hamster.main(comm, conf, rank=rank, nranks=nranks, num_nodes=num_nodes)
total_runtime = MPI.Wtime() - begin_time

if rank == 0
    open("hamster.out", "a") do file
        println(file, @sprintf("\nFinished run in %.2f seconds. Exiting Hamster...", total_runtime))
    end
end

MPI.Finalize()