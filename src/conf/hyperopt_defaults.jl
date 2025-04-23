"""
    params=[""]

The `params` tag sets the hyperparameters that are to be optimized
"""
function get_hyperopt_params(conf::Config)::Vector{String} 
    if conf("params", "HyperOpt") == "default"
        return String[] 
    else 
        if conf("params", "HyperOpt") isa Vector
            return conf("params", "HyperOpt")
        else
            return [conf("params", "HyperOpt")]
        end
    end
end

"""
    lowerbounds=[0]

The `lowerbounds` define the lower bounds of the search space.
"""
function get_hyperopt_lowerbounds(conf::Config)::Vector{Float64}
    if conf("lowerbounds", "HyperOpt") == "default" 
        return [0]
    else
        if conf("lowerbounds", "HyperOpt") isa Vector
            return conf("lowerbounds", "HyperOpt")
        else
            return [conf("lowerbounds", "HyperOpt")]
        end
    end
end

"""
    upperbounds=[0]

The `upperbounds` define the upper bounds of the search space.
"""
function get_hyperopt_upperbounds(conf::Config)::Vector{Float64}
    if conf("upperbounds", "HyperOpt") == "default" 
        return [0]
    else
        if conf("upperbounds", "HyperOpt") isa Vector
            return conf("upperbounds", "HyperOpt")
        else
            return [conf("upperbounds", "HyperOpt")]
        end
    end
end


"""
    stepsizes=[1e-5]

The `stepsizes` defines the minimum difference between two points that are sampled as trial hyperparameters.
"""
function get_hyperopt_stepsizes(conf::Config)::Vector{Float64}
    Nparams = length(get_hyperopt_params(conf))
    if conf("stepsizes", "HyperOpt") == "default" 
        return ones(Nparams)*1e-5
    else
        if conf("stepsizes", "HyperOpt") isa Vector
            return conf("stepsizes", "HyperOpt")
        else
            return [conf("stepsizes", "HyperOpt")]
        end
    end
end

"""
    niter=10

The `niter` tags sets the maximum number of iterations in the hyperparameter optimization.
"""
get_hyperopt_niter(conf::Config)::Int64 = conf("niter", "HyperOpt") == "default" ? 1 : conf("niter", "HyperOpt")

"""
    mode=random

The `mode` tag determines by which method new hyperparameter pairs are generated. Default is 'random'.
"""
get_hyperopt_mode(conf::Config)::String = conf("mode", "HyperOpt") == "default" ? "random" : conf("mode", "HyperOpt")