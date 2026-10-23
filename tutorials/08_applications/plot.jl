# Band edges (hconf_edges) and DOS, PDOS and COHP (hconf_dos) of the large cell. Run in this folder after both runs:
#   julia --project=<Hamster> plot.jl
include(joinpath(@__DIR__, "..", "plotting.jl"))
band_edges_sparse("Es.dat", eig_target=3.2)
plot_dos("dos.h5", out="dos.png", xlims=(-14, 8))
