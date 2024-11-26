"""
    Nconf=10

The `Nconf` tag sets the number of samples that are sampled from the total number of configuration.
"""
get_Nconf(conf::Config)::Int64 = get(conf, "Nconf", "Supercell", 1)

"""
    Nconf_min=10

The `Nconf_min` tag sets the minimum index that can be sampled from the total number of configurations.
"""
get_Nconf_min(conf::Config)::Int64 = get(conf, "Nconf_min", "Supercell", 1)

"""
    Nconf_max=1

The `Nconf_max` tag sets the maximum index that can be sampled from the total number of configurations.
"""
get_Nconf_max(conf::Config)::Int64 = get(conf, "Nconf_max", "Supercell", 1)

"""
    sc_poscar=SC_POSCAR

The `sc_poscar` tag sets the path to the POSCAR file used for the supercell.
"""
get_sc_poscar(conf::Config)::String = conf("POSCAR", "Supercell") == "default" ? "SC_POSCAR" : conf("POSCAR", "Supercell")

"""
    xdatcar=XDATCAR

The `xdatcar` tag sets the path to the XDATCAR file.
"""
get_xdatcar(conf::Config)::String = conf("XDATCAR", "Supercell") == "default" ? "XDATCAR" : conf("XDATCAR", "Supercell")