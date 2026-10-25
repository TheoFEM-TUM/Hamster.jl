# Validation loss of every trial of the hyperparameter optimization. Run in this folder after `hamster`:
#   julia --project=<Hamster> plot.jl
include(joinpath(@__DIR__, "..", "plotting.jl"))
plot_hyperopt("hamster_out.h5", x="Ga_alpha", y="As_alpha", out="hyperopt.png")
