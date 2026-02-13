# ====================
# ML defaults
# ====================
@configtag model Bool haskey(conf,"ML") "whether ML model is used." "ML"
@configtag update Bool decide_ml_update(conf) "optimization of ML parameters" "ML"
@configtag init_params String "zeros" "initialization of ML parameters." "ML"
@configtag filename String "ml_params" "name of ML parameter file." "ML"
@configtag mode String ifelse(get_ml_update(conf),"refit","eval") "ml training mode (eval=evaluation, refit=retrain from scratch, expand=expand on existing)" "ML"
@configtag rcut Float64 get_rcut(conf) "ml cutoff radius. Defaults to same as TB." "ML"
@configtag npoints Int64 1 "number of kernel support points." "ML"
@configtag ncluster Int64 1 "number of cluster used in kmeans clustering." "ML"
@configtag env_scale Float64 1.0 "scaling factor for environment features." "ML"
@configtag strc_scale Float64 1.0 "scaling factor for structural features." "ML"
@configtag sim_params Float64 0.1 "similarity parameter for kernel model." "ML"
@configtag sampling String "random" "sampling method for each cluster of descriptors (random, farthest)." "ML"
@configtag apply_distortion Bool false "whether distortions are considered in descriptor." "ML"
@configtag apply_distance_distortion Bool false "whether distortions (distance only) are considered in descriptor." "ML"

function decide_ml_update(conf::Config)::Bool
    if haskey(conf, "Optimizer") && haskey(conf, "ML")
        return true
    else
        return false
    end
end