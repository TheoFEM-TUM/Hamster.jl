# ====================
# Ewald defaults
# ====================
"""
    ewald=false

A model to account for long-term electrostatics (Ewald summation) is constructed if a block `Ewald` is present in the config file.
"""
get_ewald(conf::Config)::Bool = haskey(conf.blocks, "Ewald") || (conf("ewald") ≠ "default" ? conf("ewald") : false)

"""
    update=false

The `update` tag in the `Ewald` block determines whether the Ewald scaling parameters are updated during optimization.
"""
get_ewald_update(conf::Config)::Bool = conf("update", "Ewald") == "default" ? false : conf("update", "Ewald")

"""
    auto_qeff=false

If `auto_qeff = true` in the `Ewald` block, ion types without an explicit `qeff` tag get a default effective charge
based on their group in the periodic table (see `get_auto_qeff`).
"""
get_ewald_auto_qeff(conf::Config)::Bool = conf("auto_qeff", "Ewald") == "default" ? false : conf("auto_qeff", "Ewald")

"""
    method=ewald

The `method` tag in the `Ewald` block sets the method used to compute the Ewald sum.
"""
get_ewald_method(conf::Config)::String = conf("method", "Ewald") == "default" ? "ewald" : string(conf("method", "Ewald"))

"""
    rcut=rcut

The `rcut` tag in the `Ewald` block sets the cut-off radius for the real-space Ewald sum (defaults to the global `rcut`).
"""
get_ewald_rcut(conf::Config)::Float64 = conf("rcut", "Ewald") == "default" ? get_rcut(conf) : Float64(conf("rcut", "Ewald"))

"""
    charge_scale=1.0

The `charge_scale` tag in the `Ewald` block sets the initial scaling factor for the charge values.
"""
get_ewald_charge_scale(conf::Config)::Float64 = conf("charge_scale", "Ewald") == "default" ? 1.0 : Float64(conf("charge_scale", "Ewald"))

"""
    alpha=0.0

The `alpha` tag in the `Ewald` block sets the Ewald dampening parameter.
"""
get_ewald_alpha(conf::Config)::Float64 = conf("alpha", "Ewald") == "default" ? 0.0 : Float64(conf("alpha", "Ewald"))

"""
    mesh_spacing=0.5

The `mesh_spacing` tag in the `Ewald` block sets the grid spacing for reciprocal-space PME.
"""
get_ewald_mesh_spacing(conf::Config)::Float64 = conf("mesh_spacing", "Ewald") == "default" ? 0.5 : Float64(conf("mesh_spacing", "Ewald"))
