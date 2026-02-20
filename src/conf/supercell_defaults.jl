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
    config_inds=none

The `config_inds` tag sets a file from which configuration indices are read. By default, indices are not read from file.
It is also possible to provide a list of integers directly.
"""
function get_config_inds(conf::Config)::Union{String, Vector{Int64}}
    if conf("config_inds", "Supercell") == "default" 
        return "none" 
    else 
        config_inds = conf("config_inds", "Supercell")
        if config_inds isa Int64
            return [config_inds]
        elseif config_inds isa Vector{Int64}
            return config_inds
        else
            return string(config_inds)
        end
    end
end
"""
    config_inds=none

The `config_inds` tag sets a file from which configuration indices are read. By default, indices are not read from file.
It is also possible to provide a list of integers directly.
"""
function get_val_config_inds(conf::Config)::Union{String, Vector{Int64}}
    if conf("val_config_inds", "Supercell") == "default" 
        return "none" 
    else 
        config_inds = conf("val_config_inds", "Supercell")
        if config_inds isa Int64
            return [config_inds]
        elseif config_inds isa Vector{Int64}
            return config_inds
        else
            return string(config_inds)
        end
    end
end
"""
    nbatch=1

The `nbatch` tag detemines into how many batches the training structures are split for stochastic gradient optimization.
"""
get_nbatch(conf::Config)::Int64 = conf("nbatch", "Supercell") == "default" ? 1 : conf("nbatch", "Supercell")

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

"""
    xdatcar_val=XDATCAR_val

The `xdatcar_val` tag sets the path to the validation XDATCAR file.
"""
get_xdatcar_val(conf::Config)::String = conf("XDATCAR_val", "Supercell") == "default" ? "XDATCAR_val" : conf("XDATCAR_val", "Supercell")