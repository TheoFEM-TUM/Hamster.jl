# Loss curve of the SOC fit and the band structure with TB + SOC. Run in the folder of the fit after run.sh steps:
#   julia --project=<Hamster> ../plot.jl
include(joinpath(@__DIR__, "..", "plotting.jl"))
plot_losses("hamster_out_fit.h5", out="losses.png")
plot_bands("Es.dat", reference=joinpath(DATA, "cspbbr3", "EIGENVAL_soc"), nshift=26, ylims=(-4, 6),
           title="CsPbBr3: TB + SOC vs DFT", out="bands.png")
