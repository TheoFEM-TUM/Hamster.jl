# ====================
# ML defaults
# ====================
@configtag model Bool haskey(conf,"ML") "whether ML model is used." "ML" category="model" default_doc="true if an ML block is present"
@configtag update Bool decide_ml_update(conf) "optimization of ML parameters" "ML" category="model" default_doc="true in a fit (Optimizer and ML blocks), else false"
@configtag init_params String "zeros" "initialization of ML parameters." "ML" category="parameters"
@configtag filename String "ml_params" "name of ML parameter file." "ML" category="parameters"
@configtag mode String ifelse(get_ml_update(conf),"refit","eval") "ml training mode (eval=evaluation, refit=retrain from scratch, expand=expand on existing)" "ML" category="model" default_doc="refit if update = true, else eval"
@configtag rcut Float64 get_rcut(conf) "ml cutoff radius. Defaults to same as TB." "ML" category="descriptor" default_doc="same as rcut of the Options block"
@configtag npoints Int64 1 "number of kernel support points." "ML" category="data points" see_also=["ncluster", "sampling"]
@configtag ncluster Int64 1 "number of cluster used in kmeans clustering." "ML" category="data points" see_also=["npoints"]
@configtag env_scale Float64 1.0 "scaling factor for environment features." "ML" category="descriptor"
@configtag strc_scale Float64 1.0 "scaling factor for structural features." "ML" category="descriptor"
@configtag sim_params Float64 0.1 "similarity parameter for kernel model." "ML" category="model"
@configtag sampling String "random" "sampling method for each cluster of descriptors (random, farthest)." "ML" category="data points"
@configtag apply_distortion Bool false "whether distortions are considered in descriptor." "ML" category="descriptor"
@configtag apply_distance_distortion Bool false "whether distortions (distance only) are considered in descriptor." "ML" category="descriptor"
@configtag apply_orthogonality Bool false "enforce orthogonality of atomic orbitals of different l, m (r=0)" "ML" category="descriptor"
@configtag sim_mat Bool false "precalc similarity matrix of ML descriptors." "ML" category="precomputed similarities" see_also=["sim_tol"]
@configtag sim_tol Float64 0.01 "cutoff tolerance for similarity matrix." "ML" category="precomputed similarities"
@configtag key_dims Vector{Int64} Int64[] "Which descriptor dimensions to be used as descriptor keys. Only use discrete dimensions!" "ML" category="descriptor"

function decide_ml_update(conf::Config)::Bool
    if haskey(conf, "Optimizer") && haskey(conf, "ML")
        return true
    else
        return false
    end
end