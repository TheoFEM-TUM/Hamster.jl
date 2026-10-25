# Compare the band structure of the run (Es.dat) with the DFT bands. Run after `hamster` in this folder.
include(joinpath(@__DIR__, "..", "plotting.jl"))
cd(@__DIR__)
plot_bands("Es.dat", reference="../data/gaas/EIGENVAL_bands", title="GaAs: TB model vs DFT", out="bands.png")
