# ====================
# HyperOpt defaults
# ====================
@configtag params Vector{String} String[] "hyperparameters to be optimized as [BLOCK_]PARAM." "HyperOpt"
@configtag lowerbounds Vector{Float64} [0.] "lower bounds of the search space." "HyperOpt"
@configtag upperbounds Vector{Float64} [0.] "upper bounds of the search space." "HyperOpt"
@configtag stepsizes Vector{Float64} ones(length(get_hyperopt_params(conf)))*1e-5 "minimum difference between two hyperparameter samples" "HyperOpt"
@configtag niter Int64 1 "number of hyperparameter optimization steps." "HyperOpt"
@configtag mode String "random" "hyperparameter optimization strategy (random=random search, grid=grid search, tpe=tree-structured Parzen Estimator)" "HyperOpt"