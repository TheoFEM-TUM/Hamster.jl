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