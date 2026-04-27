# ====================
# Ewald defaults
# ====================
"""
    ewald=false

A model to account for long-term electrostatics (Ewald summation) is constructed if a block `Ewald` is present in the config file.
"""
get_ewald(conf::Config)::Bool = haskey(conf.blocks, "Ewald") || (conf("ewald") ≠ "default" ? conf("ewald") : false)
push!(CONFIG_TAGS, ConfigTag{Bool}("ewald", conf->get_ewald(conf), "activates ewald model."))

@configtag update Bool false "update Ewald scaling parameter" "Ewald"
@configtag method String "pme" "method to compute the Ewald sum" "Ewald"
@configtag charge_scale Float64 1.0 "scaling factor for charge values" "Ewald"