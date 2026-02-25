# ====================
# Supercell defaults
# ====================
@configtag Nconf Int64 1 "number of sample configuration." "Supercell"
@configtag Nconf_min Int64 1 "minimum configuration index for sample." "Supercell"
@configtag Nconf_max Int64 1 "minimum configuration index for sample." "Supercell"
@configtag nbatch Int64 1 "number of batches to split the data set." "Supercell"


"""
*sc_poscar*=SC_POSCAR

The `sc_poscar` tag sets the path to the POSCAR file used for the supercell. Only accepts VASP POSCAR format
"""
get_sc_poscar(conf::Config)::String = conf("POSCAR", "Supercell") == "default" ? "SC_POSCAR" : conf("POSCAR", "Supercell")
push!(CONFIG_TAGS, ConfigTag{String}("poscar", "Supercell", conf->get_sc_poscar(conf), "path to supercell structure file."))

"""
*xdatcar* = XDATCAR

Specify the path to the structural trajectory used in the calculation.

The `xdatcar` tag points to a file containing atomic positions and lattice information for a sequence of structures.

# Accepted Formats

- **VASP `XDATCAR` file**  
  Parsed directly to obtain atomic positions and lattice vectors for each structure.

- **HDF5 (`.h5`) file**  
  Must contain the datasets:
  - `positions` — atomic positions with shape **[3 × N_atoms × N_structures]**  
  - `lattice` — lattice vectors with shape **[3 × 3 × N_structures]** (optional for each structure)
"""
get_xdatcar(conf::Config)::String = conf("XDATCAR", "Supercell") == "default" ? "XDATCAR" : conf("XDATCAR", "Supercell")
push!(CONFIG_TAGS, ConfigTag{String}("xdatcar", conf->get_xdatcar(conf), "path to supercell trajectory file."))

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
    val_config_inds=none

The `val_config_inds` tag sets a file from which configuration indices are read. By default, indices are not read from file.
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