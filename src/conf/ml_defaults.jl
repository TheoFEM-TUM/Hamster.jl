"""
**init_params**=zeros

The `init_params` tag determines how the parameters of the ML model are initialized.
"""
get_ml_init_params(conf::Config)::String = get(conf, "init_params", "ML", "zeros")

"""
**filename**=ml_params

The `filename` tag sets the name for the parameter file of the ML model.
"""
get_ml_filename(conf::Config)::String = get(conf, "filename", "ML", "ml_params")

"""
    ml_model=false

The `ml_model` tag switches on the use of an ML model in the effective Hamiltonian.
"""
get_ml_model(conf::Config)::Bool = haskey(conf, "ML")

"""
    update=true

The `update` tag switches on/off optimization of ML parameters.
"""
function get_ml_update(conf::Config)::Bool
    if conf("update", "ML") == "default" && haskey(conf, "Optimizer") && haskey(conf, "ML")
        return true
    elseif conf("update", "ML") == "default" && (!haskey(conf, "Optimizer") || !haskey(conf, "ML"))
        return false
    else 
        return conf("update", "ML")
    end
end

"""
**mode** = "eval"

Training mode for the ML model.

Possible values:
- `"eval"`   : Evaluation mode. No training is performed.
- `"refit"`  : Train (or retrain) the model from scratch.
- `"expand"` : Load existing parameters from `filename`, append newly
               initialized parameters (according to `init_params`).
"""
get_ml_mode(conf)::String = get_ml_update(conf) ? get(conf, "mode", "ML", "refit") : get(conf, "mode", "ML", "eval")

"""
    ml_rcut=rcut

Sets the cut-off radius for the ML model. Defaults to the same cut-off radius as the TB model.
"""
get_ml_rcut(conf::Config)::Float64 = conf("rcut", "ML") == "default" ? get_rcut(conf) : conf("rcut", "ML")

"""
    env_scale=1.

Sets the scaling factor that is multiplied with the environmental descriptor.
"""
get_env_scale(conf::Config)::Float64 = conf("env_scale", "ML") == "default" ? 1.0 : conf("env_scale", "ML")

"""
    strc_scale=1.

Sets the scaling factor that is multiplied with the structural descriptor entries.
"""
get_strc_scale(conf::Config)::Float64 = conf("strc_scale", "ML") == "default" ? 1.0 : conf("strc_scale", "ML")

"""
    sim_params=0.1

Sets the parameter for the similarity function of the kernel model.
"""
get_sim_params(conf::Config)::Float64 = conf("sim_params", "ML") == "default" ? 0.1 : conf("sim_params", "ML")

"""
    sampling=random

The `sampling` tag determines how points are selected from each cluster. Defaults to "random".
"""
get_ml_sampling(conf::Config)::String = conf("sampling", "ML") == "default" ? "random" : conf("sampling", "ML")

"""
    apply_distortion=false

If true, distortions are applied to atomic positions in calculating descriptor values.
"""
get_apply_distortion(conf::Config)::Bool = conf("apply_distortion", "ML") == "default" ? false : conf("apply_distortion", "ML")

"""
    apply_distance_distortion=false

If true, distortions are applied to atomic positions in calculating atomic distance in ML descriptor.
"""
get_apply_distance_distortion(conf::Config)::Bool = conf("apply_distance_distortion", "ML") == "default" ? false : conf("apply_distance_distortion", "ML")

"""
    ncluster=1

Sets the number of clusters to be used in the kmeans clustering.
"""
get_ml_ncluster(conf::Config)::Int64 = conf("Ncluster", "ML") == "default" ? 1 : conf("Ncluster", "ML")

"""
    npoints=1

Sets the number of data points that are sampled in total.
"""
get_ml_npoints(conf::Config)::Int64 = conf("Npoints", "ML") == "default" ? 1 : conf("Npoints", "ML")