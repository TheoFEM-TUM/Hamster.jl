"""
    soc=false

A model to account for spin-orbit coupling (SOC) is constructed if a block `SOC` is present in the config file.
"""
get_soc(conf::Config)::Bool = haskey(conf.blocks, "SOC") || (conf("soc") ≠ "default" ? conf("soc") : false)
push!(CONFIG_TAGS, ConfigTag{Bool}("soc", conf->get_soc(conf), "activates soc model."))

@configtag update Bool haskey(conf, "Optimizer") && haskey(conf, "SOC") "soc parameters are updated." "SOC"

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
push!(CONFIG_TAGS, ConfigTag{String}("init_params", "SOC", conf->get_soc_init_params(conf), "soc parameter initialization."))