"""
    params=[""]

The `params` tag sets the hyperparameters that are to be optimized
"""
get_hyperopt_params(conf::Config)::Vector{String} = conf("params", "HyperOpt") == "default" ? [""] : conf("params", "HyperOpt")

"""
    lowerbounds=[0]

The `lowerbounds` define the lower bounds of the search space.
"""
get_hyperopt_lowerbounds(conf::Config)::Vector{Float64} = conf("lowerbounds", "HyperOpt") == "default" ? [0] : conf("lowerbounds", "HyperOpt")

"""
    upperbounds=[0]

The `upperbounds` define the upper bounds of the search space.
"""
get_hyperopt_upperbounds(conf::Config)::Vector{Float64} = conf("upperbounds", "HyperOpt") == "default" ? [0] : conf("upperbounds", "HyperOpt")

"""
    niter=10

The `niter` tags sets the maximum number of iterations in the hyperparameter optimization.
"""
get_hyperopt_niter(conf::Config)::Int64 = conf("niter", "HyperOpt") == "default" ? 10 : conf("niter", "HyperOpt")