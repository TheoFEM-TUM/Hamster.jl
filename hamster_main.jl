using MPI, Printf

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Hamster is loaded on rank 0 first. If it has to be precompiled (e.g. after an update or on a node with another CPU
# type), only rank 0 does so, and the other ranks load the finished cache instead of all compiling at the same time.
rank == 0 && @eval using Hamster
MPI.Barrier(comm)
rank ≠ 0 && @eval using Hamster

num_nodes = parse(Int64, get(ENV, "SLURM_JOB_NUM_NODES", "1"))

args = Hamster.parse_commandline(ARGS)

conf = haskey(args, "conf") ? get_config(filename=args["conf"]) : get_config()
for (key, value) in args
    set_value!(conf, key, value)
end
if rank ≠ 0; set_value!(conf, "verbosity", 0); end

begin_time = MPI.Wtime()
Hamster.main(comm, conf, rank=rank, nranks=nranks, num_nodes=num_nodes)
total_runtime = MPI.Wtime() - begin_time

if rank == 0
    open("hamster.out", "a") do file
        println(file, @sprintf("\nFinished run in %.2f seconds. Exiting Hamster...", total_runtime))
    end
end

MPI.Finalize()