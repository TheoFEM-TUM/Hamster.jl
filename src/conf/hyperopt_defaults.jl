# ====================
# HyperOpt defaults
# ====================
@configtag params Vector{String} String[] "hyperparameters to be optimized: tag (Options block or a tag of only one block), Block_tag (e.g. ML_rcut) or Element_tag (e.g. Ga_alpha)." "HyperOpt" category="search"
@configtag lowerbounds Vector{Float64} [0.] "lower bounds of the search space." "HyperOpt" category="search"
@configtag upperbounds Vector{Float64} [0.] "upper bounds of the search space." "HyperOpt" category="search"
@configtag stepsizes Vector{Float64} ones(length(get_hyperopt_params(conf)))*1e-5 "minimum difference between two hyperparameter samples" "HyperOpt" category="search" default_doc="1e-5 for each parameter"
@configtag niter Int64 1 "number of candidates of random search and TPE (grid search evaluates all grid points)." "HyperOpt" category="search"
@configtag mode String "random" "hyperparameter optimization strategy (random=random search, grid=grid search, tpe=tree-structured Parzen Estimator)" "HyperOpt" category="search" values=["random", "grid", "tpe"]