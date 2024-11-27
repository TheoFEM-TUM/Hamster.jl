function main(conf; num_nodes=1)
    nhamster = get_nhamster(conf)
    hostnames = pmap(1:nworkers()) do i
        readchomp(`hostname`)
    end
    generate_output(conf, hostnames=hostnames)
    nthreads_kpoints = pmap(worker->get_nthreads_kpoints(conf), workers())
    nthreads_bands = pmap(worker->get_nthreads_bands(conf), workers())
    write_block_summary("Parallelization", num_nodes=num_nodes, nhamster_per_node=nhamster, nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands)

    task = decide_which_task_to_perform(conf)
    run_calculation(task, conf)
end

"""
    decide_which_task_to_perform(conf::Config)

Given a `Config` instance, decides which type of calculation is to be performed based on certain tags in the input file.

# Details
- An `Optimizer` block tells Hamster to run a parameter optimization.
"""
function decide_which_task_to_perform(conf::Config)
    if haskey(conf, "Optimizer")
        return Val{:optimization}()
    end
    error("Your given configuration does not specify which calculation to perform. Check your input!")
end