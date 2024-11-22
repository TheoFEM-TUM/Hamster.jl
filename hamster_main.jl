using Pkg; Pkg.add("ClusterManagers")
using Distributed, Hamster, ClusterManagers

args = Hamster.parse_commandline(ARGS)

conf = haskey(args, "conf") ? get_config(filename=args["conf"]) : get_config()
for (key, value) in args
    set_value!(conf, key, value)
end
nhamster = Hamster.get_nhamster(conf)
num_nodes = parse(Int64, get(ENV, "SLURM_JOB_NUM_NODES", "1"))

if haskey(ENV, "SLURM_JOB_NODELIST")
    nodelist = split(ENV["SLURM_JOB_NODELIST"], ',')
    nworker_per_node = floor(Int64, nhamster / length(nodelist))
    @show nworker_per_node
    redirect_stdout(Base.DevNull()) do
        addprocs(SlurmManager(96), distribution="cyclic", output="hamster.out")
    end
else
    addprocs(nhamster)
end

@everywhere using Hamster

Hamster.main(ARGS, conf)

rmprocs(workers())