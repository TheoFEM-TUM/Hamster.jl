"""
    soc=false

A model to account for spin-orbit coupling (SOC) is constructed if a block `SOC` is present in the config file.
"""
get_soc(conf::Config)::Bool = haskey(conf.blocks, "SOC") || (conf("soc") ≠ "default" ? conf("soc") : false)

"""
    update_soc=true

The `update_soc` tag determines whether the SOC parameters are udpated.
"""
get_update_soc(conf::Config)::Bool = conf("update", "SOC") == "default" ? true : conf("update", "SOC")

"""
    init_params=zeros

The `init_params` tag determines how SOC parameters are initialized.
"""
get_soc_init_params(conf::Config)::String = conf("init_params", "SOC") == "default" ? "zeros" : conf("init_params", "SOC")