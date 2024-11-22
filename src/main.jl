function main(args_list)
    args = parse_commandline(args_list)
    
    hostname = readchomp(`hostname`)
    user = ENV["USER"]
    @show hostname, user

    conf = haskey(args, "conf") ? get_config(filename=args["conf"]) : get_config()
    for (key, value) in args
        set_value!(conf, key, value)
    end
    nhamster = get_nhamster(conf)
    nworker_per_node = floor(Int64, nhamster / length([hostname]))
    for node in [hostname]
        addprocs([("$user@$node", nworker_per_node)])
    end
    @show nworkers()
    rmprocs(workers())
end