function main(args_list, conf)
    nhamster = get_nhamster(conf)
    hostnames = pmap(1:nworkers()) do i
        readchomp(`hostname`)
    end
    @show hostnames
end