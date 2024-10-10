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