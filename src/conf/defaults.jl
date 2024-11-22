"""
    verbosity=1

The `verbosity` parameter sets the quanitity of the output that is printed to the console or written to output files. A value of 0 deactivates all print statements.
"""
get_verbosity(conf::Config)::Int64 = conf("verbosity") == "default" ? 1 : conf("verbosity")

"""
    system=unknown

The `system` tag gives a name to the system under study.
"""
get_system(conf::Config)::String = conf("system") == "default" ? "unknown" : conf("system")

"""
    init_params=ones

The `init_params` tag determines how the TB parameters are initialized. Possible optiones are `ones`, `random` or a file of `name`.
"""
get_init_params(conf::Config)::String = conf("init_params") == "default" ? "ones" : conf("init_params")

"""
    nthreads_kpoints=JULIA_NUM_THREADS

The `nthreads_kpoints` tag sets the number of tasks to work on kpoints simultaneously.
"""
get_nthreads_kpoints(conf::Config)::Int64 = conf("nthreads_kpoints") == "default" ? Threads.nthreads() : conf("nthreads_kpoints") 

"""
    nthreads_bands=JULIA_NUM_THREADS

The `nthreads_bands` tag sets the number of tasks to work on energy bands simultaneously.
"""
get_nthreads_bands(conf::Config)::Int64 = conf("nthreads_bands") == "default" ? Threads.nthreads() : conf("nthreads_bands")

"""
    nhamster=1

The `nhamster` tag sets the number of `Hamster` processes to be spawned for parallel tasks.
"""
get_nhamster(conf::Config)::Int64 = conf("nhamster") == "default" ? 1 : conf("nhamster")