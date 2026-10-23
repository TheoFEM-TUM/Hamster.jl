# Training and validation loss of the electrostatics fit. At iteration 1, the couplings are zero (charge_scale = 0): the
# loss there is the error of the TB model alone on the MD snapshots. Run in the folder of the fit after `hamster`:
#   julia --project=<Hamster> ../plot.jl
include(joinpath(@__DIR__, "..", "plotting.jl"))
plot_losses("hamster_out.h5", out="losses.png")
