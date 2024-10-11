"""
    update_tb=haskey(conf, "Optimizer")

The `update_tb` tag determines whether the parameters of the TB model are updated during the optimization. Defaults to `true` if an `Optimizer` block is provided.
"""
get_update_tb(conf::Config)::Bool = conf("update_tb", "Optimizer") == "default" ? haskey(conf, "Optimizer") : conf("update_tb", "Optimizer")

"""
    loss=MAE

The `loss` tag sets the loss function to be used for the optimization.
"""
get_loss(conf::Config)::String = conf("loss", "Optimizer") == "default" ? "MAE" : conf("loss", "Optimizer")

"""
    wE=ones

The `wE` tag sets the weight of each energy band for the calculation of the loss. Individual weights can also be set with, e.g., `wE_3 = 2`.
"""
get_band_weights(conf::Config, Nε)::Vector{Float64} = conf("wE", "Optimizer") == "default" ? ones(Nε) : conf("wE", "Optimizer")

"""
    wk=ones

The `wk` tag sets the weight of each kpoint for the calculation of the loss. Individual weights can also be set with, e.g., `wk_3 = 2`.
"""
get_kpoint_weights(conf::Config, Nk)::Vector{Float64} = conf("wk", "Optimizer") == "default" ? ones(Nk) : conf("wk", "Optimizer")

"""
    lreg=2

The `lreg` tag determines which norm is used to calculate the regularization, e.g., L2 for `lreg=2`.
"""
get_lreg(conf::Config)::Int64 = conf("lreg", "Optimizer") == "default" ? 2 : conf("lreg", "Optimizer")

"""
    lambda=0.

The `lambda` parameter determines the regularization constant.
"""
get_lambda(conf::Config)::Float64 = conf("lambda", "Optimizer") == "default" ? 0. : conf("lambda", "Optimizer")

"""
    barrier=0.

The `barrier` parameter determines at which magnitude the regularization kicks in.
"""
get_barrier(conf::Config)::Float64 = conf("barrier", "Optimizer") == "default" ? 0. : conf("barrier", "Optimizer")