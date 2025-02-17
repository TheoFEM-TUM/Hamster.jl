"""
    init_params=zeros

The `init_params` tag determines how the parameters of the ML model are initialized.
"""
get_ml_init_params(conf::Config)::String = conf("init_params", "ML") == "default" ? "zeros" : conf("init_params", "ML")

"""
    filename=ml_params

The `filename` tag sets the name for the parameter file of the ML model.
"""
get_ml_filename(conf::Config)::String = conf("filename", "ML") == "default" ? "ml_params" : conf("filename", "ML")

"""
    ml_model=false

The `ml_model` tag switches on the use of an ML model in the effective Hamiltonian.
"""
get_ml_model(conf::Config)::Bool = haskey(conf, "ML")

"""
    ml_optimize=false

The `ml_optimize` tag switches on optimization of the ML model.
"""
get_ml_optimize(conf::Config)::Bool = conf("optimize", "ML") == "default" ? false : conf("optimize", "ML")

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
    sim_params=0.1

Sets the parameter for the similarity function of the kernel model.
"""
get_sim_params(conf::Config)::Float64 = conf("sim_params", "ML") == "default" ? 0.1 : conf("sim_params", "ML")

"""
    apply_distortion=false

If true, distortions are applied to atomic positions in calculating descriptor values.
"""
get_apply_distortion(conf::Config)::Bool = conf("apply_distortion", "ML") == "default" ? false : conf("apply_distortion", "ML")

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