"""
    soc=false

A model to account for spin-orbit coupling (SOC) is constructed if a block `SOC` is present in the config file.
"""
get_soc(conf::Config)::Bool = haskey(conf.blocks, "SOC") || (conf("soc") ≠ "default" ? conf("soc") : false)
@configtag soc Bool false "spin-orbit coupling model (also switched on by a SOC block)." getter=:custom default_doc="true if a SOC block is present, else false" category="model"

@configtag update Bool haskey(conf, "Optimizer") && haskey(conf, "SOC") "soc parameters are updated." "SOC" category="model" default_doc="true in a fit (Optimizer and SOC blocks), else false"

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
@configtag init_params String "zeros" "initialization of the SOC parameters (file, ones, zeros, rand)." "SOC" getter=:custom default_doc="zeros (in a standard run: init_params of the Options block if set)" category="model"