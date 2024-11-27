function main(comm, conf; rank=0, nranks=1, num_nodes=1)
    hostnames = MPI.gather(readchomp(`hostname`), comm, root=0)
    if rank == 0
        generate_output(conf, hostnames=hostnames)
        nthreads_kpoints = get_nthreads_kpoints(conf)
        nthreads_bands = get_nthreads_bands(conf)
        write_block_summary("Parallelization", num_nodes=num_nodes, nhamster=nranks, nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands)
    end

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