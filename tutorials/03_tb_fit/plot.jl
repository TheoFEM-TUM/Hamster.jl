# Loss curves of the fit and the band structure with the fitted parameters. Run in the folder of the fit
# (your own folder or solution/) after `hamster` and `hamster --conf hconf_bands`:  julia --project=<Hamster> ../plot.jl
include(joinpath(@__DIR__, "..", "plotting.jl"))
plot_losses("hamster_out_fit.h5", out="losses.png")
plot_bands("Es.dat", reference=joinpath(DATA, "gaas", "EIGENVAL_bands"), title="GaAs: fitted TB model vs DFT", out="bands.png")
