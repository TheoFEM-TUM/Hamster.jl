# ====================
# ML defaults
# ====================
@configtag model Bool haskey(conf,"ML") "whether ML model is used." "ML" category="model" default_doc="true if an ML block is present"
@configtag update Bool decide_ml_update(conf) "optimization of ML parameters" "ML" category="model" default_doc="true in a fit (Optimizer and ML blocks), else false"
@configtag init_params String "zeros" "initialization of ML parameters." "ML" category="parameters"
@configtag filename String "ml_params" "name of ML parameter file." "ML" category="parameters"
@configtag mode String ifelse(get_ml_update(conf),"refit","eval") "ml training mode (eval=evaluation, refit=retrain from scratch, expand=expand on existing)" "ML" category="model" default_doc="refit if update = true, else eval" values=["eval", "refit", "expand"]
@configtag rcut Float64 get_rcut(conf) "ml cutoff radius. Defaults to same as TB." "ML" category="descriptor" default_doc="same as rcut of the Options block"
@configtag npoints Int64 1 "number of kernel support points." "ML" category="data points" see_also=["ncluster", "sampling"]
@configtag ncluster Int64 1 "number of cluster used in kmeans clustering." "ML" category="data points" see_also=["npoints"]
@configtag env_scale Float64 1.0 "scaling factor for environment features." "ML" category="descriptor"
@configtag strc_scale Float64 1.0 "scaling factor for structural features." "ML" category="descriptor"
@configtag sim_params Float64 0.1 "similarity parameter for kernel model." "ML" category="model"
@configtag sampling String "random" "sampling method for each cluster of descriptors: random, farthest (farthest-point sampling) or uncertainty (greedy largest GP posterior variance w.r.t. the already selected points); r, f and u for short." "ML" category="data points" values=["random", "farthest", "r", "f", "uncertainty", "u"]
@configtag unc_noise Float64 1e-3 "noise level λ of the Gaussian-process posterior variance (prediction uncertainty of the kernel model); shifts the scale of the variance, not the ranking." "ML" category="uncertainty" see_also=["sampling", "unc_threshold", "uncertainty"]
@configtag unc_threshold Float64 0.5 "the model summary warns about out-of-distribution predictions if the mean prediction uncertainty of a structure exceeds this value (GP posterior variance, between 0 and 1)." "ML" category="uncertainty" see_also=["unc_noise", "unc_rel", "uncertainty"]
@configtag unc_rel Float64 2.0 "with at least two structures in a run, the model summary also warns about structures whose mean prediction uncertainty exceeds unc_rel times the median over the structures of the run (relative to the training structures in a fit)." "ML" category="uncertainty" see_also=["unc_threshold", "uncertainty"]
@configtag unc_max_desc Int64 10000 "maximum number of descriptor vectors per structure used to estimate the prediction uncertainty in the model summary (a random subset if there are more); bounds the cost, which scales with the square of the number of data points." "ML" category="uncertainty" see_also=["uncertainty"]
@configtag uncertainty Bool true "whether the prediction uncertainty (GP posterior variance of the kernel model) is computed, written to the model summary and available via get_uncertainty." "ML" category="uncertainty" see_also=["unc_noise", "unc_threshold", "unc_max_desc"]
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
