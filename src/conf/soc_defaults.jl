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
function get_soc_init_params(conf::Config)::String
    soc_init = conf("init_params", "SOC")
    tb_init = conf("init_params")
    if soc_init == "default" && get_soc(conf) && tb_init ≠ "default" && !haskey(conf.blocks, "Optimizer")
        return get_init_params(conf)
    elseif soc_init == "default" && get_soc(conf)
        return "zeros"
    else
        return soc_init
    end
end