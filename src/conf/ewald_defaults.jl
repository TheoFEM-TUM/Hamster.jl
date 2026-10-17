# ====================
# Ewald defaults
# ====================
"""
    ewald=false

A model to account for long-term electrostatics (Ewald summation) is constructed if a block `Ewald` is present in the config file.
"""
get_ewald(conf::Config)::Bool = haskey(conf.blocks, "Ewald") || (conf("ewald") ≠ "default" ? conf("ewald") : false)
push!(CONFIG_TAGS, ConfigTag{Bool}("ewald", conf->get_ewald(conf), "activates ewald model."))

@configtag update Bool false "fit the Ewald coupling of each species (starting from charge_scale or init_params)." "Ewald"
@configtag init_params String "default" "file with the Ewald coupling per species (written by a fit), default: charge_scale for all species." "Ewald"
@configtag filename String "ewald_params.dat" "file to which the fitted Ewald coupling per species is written." "Ewald"
@configtag method String "ewald" "method to compute the Ewald sum" "Ewald"
@configtag rcut Float64 get_rcut(conf) "cut-off radius for real-space ewald" "Ewald"
@configtag charge_scale Float64 1.0 "scaling factor for charge values" "Ewald"
@configtag alpha Float64 0. "Ewald dampening parameter." "Ewald"
@configtag mesh_spacing Float64 0.5 "Grid spacing for reciprocal-space PME." "Ewald"