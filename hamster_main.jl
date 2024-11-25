using Pkg; Pkg.add("ClusterManagers")
using Distributed, Hamster, ClusterManagers

args = Hamster.parse_commandline(ARGS)

conf = haskey(args, "conf") ? get_config(filename=args["conf"]) : get_config()
for (key, value) in args
    set_value!(conf, key, value)
end
nhamster = Hamster.get_nhamster(conf)
num_nodes = parse(Int64, get(ENV, "SLURM_JOB_NUM_NODES", "1"))
num_threads = Int(Sys.CPU_THREADS / nhamster)

if haskey(ENV, "SLURM_JOB_NODELIST")
    nworker_total = nhamster * num_nodes
    addprocs(SlurmManager(nworker_total), distribution="cyclic", exeflags=["--project", "-t $num_threads"])
else
    addprocs(nhamster, exeflags=["--project", "-t $num_threads"])
end

@everywhere using Hamster

Hamster.main(ARGS, conf)

rmprocs(workers())