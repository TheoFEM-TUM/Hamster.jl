# ====================
# Supercell defaults
# ====================
@configtag Nconf Int64 1 "number of configurations sampled from the trajectory (between Nconf_min and Nconf_max, in addition to config_inds)." "Supercell" category="structure"
@configtag Nconf_min Int64 1 "minimum configuration index for sample." "Supercell" category="structure"
@configtag Nconf_max Int64 1 "maximum configuration index for sample (at most the number of configurations of the trajectory)." "Supercell" category="structure"
@configtag nbatch Int64 1 "number of batches per iteration of a fit (parameter updates per iteration); every MPI rank splits its training structures into this many batches, at most the largest number of training structures of a rank." "Supercell" category="structure"


"""
*sc_poscar*=SC_POSCAR

The `sc_poscar` tag sets the path to the POSCAR file used for the supercell. Only accepts VASP POSCAR format
"""
get_sc_poscar(conf::Config)::String = conf("POSCAR", "Supercell") == "default" ? "SC_POSCAR" : conf("POSCAR", "Supercell")
@configtag poscar String "SC_POSCAR" "path to the POSCAR of the supercell (the reference structure of the trajectory)." "Supercell" getter=:custom value=(conf -> get_sc_poscar(conf)) category="structure" see_also=["xdatcar"]

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
@configtag xdatcar String "XDATCAR" "path to the trajectory: a VASP XDATCAR or an HDF5 file with positions (3 × atoms × structures) and optionally lattice (3 × 3 × structures)." "Supercell" getter=:custom value=(conf -> get_xdatcar(conf)) category="structure" see_also=["poscar", "Nconf"]

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

@configtag config_inds String "none" "configurations to use: a list of indices separated by spaces, a .dat file (number of indices in the first line, then one index per line) or an HDF5 file with the dataset train_config_inds (e.g. hamster_out.h5 of an earlier fit)." "Supercell" getter=:custom value=(conf -> string(get_config_inds(conf))) default_doc="none (Nconf configurations between Nconf_min and Nconf_max)" category="structure"
@configtag val_config_inds String "none" "validation configurations: a list of indices, a .dat file or an HDF5 file with the dataset val_config_inds (formats as for config_inds)." "Supercell" getter=:custom value=(conf -> string(get_val_config_inds(conf))) default_doc="none" category="structure"

"""
    val_xdatcar=XDATCAR (same as xdatcar)

Trajectory of the validation structures of a fit, in the formats of `xdatcar` (in universal mode an HDF5 file with one
group per system; its systems may differ from those of the training set). If it is set, the validation structures come
from this file instead of the training trajectory: all its configurations, or those of `val_config_inds`, independent
of the training configurations (`Nconf`, `val_ratio`, `Nconf_min` and `Nconf_max` apply only to the training set).
Configuration `n` of the file belongs to structure `n` of `val_data`. It is needed for universal validation if the
training set is not universal (e.g. `train_mode = pc`). Earlier versions read it from the `Optimizer` block; that still
works, with a warning.
"""
get_val_xdatcar(conf::Config)::String = separate_val_structures(conf) ? conf("val_xdatcar", "Supercell") : get_xdatcar(conf)
@configtag val_xdatcar String "XDATCAR" "trajectory of the validation structures of a fit (formats as xdatcar; universal: one group per system): the validation set is then all its configurations (or val_config_inds), independent of the training set." "Supercell" getter=:custom value=(conf -> get_val_xdatcar(conf)) default_doc="same as xdatcar" category="structure" see_also=["val_poscar", "val_config_inds", "val_data"]

"""
    separate_val_structures(conf) -> Bool

Whether the validation structures come from a trajectory of their own (`val_xdatcar` is set).
"""
separate_val_structures(conf::Config)::Bool = conf("val_xdatcar", "Supercell") != "default"

"""
    val_poscar=SC_POSCAR (same as poscar)

Reference structure (POSCAR) of the validation trajectory `val_xdatcar` in md mode. Not needed in universal mode, where
the HDF5 file contains the atom types. Earlier versions read it from the `Optimizer` block; that still works, with a
warning.
"""
get_val_poscar(conf::Config)::String = conf("val_poscar", "Supercell") == "default" ? get_sc_poscar(conf) : conf("val_poscar", "Supercell")
@configtag val_poscar String "SC_POSCAR" "reference POSCAR of the validation trajectory val_xdatcar (md mode)." "Supercell" getter=:custom value=(conf -> get_val_poscar(conf)) default_doc="same as poscar" category="structure" see_also=["val_xdatcar"]

"""
    has_trajectory(conf) -> Bool

Whether `conf` has a `Supercell` block that sets more than the validation trajectory (`val_xdatcar`, `val_poscar`):
the structures of the calculation (training set of a fit) are then supercells of a trajectory.
"""
has_trajectory(conf::Config)::Bool =
    haskey(conf, "Supercell") && any(∉(("val_xdatcar", "val_poscar")), keys(conf.blocks["Supercell"]))
