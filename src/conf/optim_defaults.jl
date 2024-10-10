"""
    update_tb=haskey(conf, "Optimizer")

The `update_tb` tag determines whether the parameters of the TB model are updated during the optimization. Defaults to `true` if an `Optimizer` block is provided.
"""
get_update_tb(conf::Config)::Bool = conf("update_tb", "Optimizer") == "default" ? haskey(conf, "Optimizer") : conf("update_tb", "Optimizer")