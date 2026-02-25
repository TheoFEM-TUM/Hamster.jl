# ====================
# Structure defaults
# ====================
@configtag rcut Float64 7.0 "Distance beyond which all interactions are neglected."
@configtag rcut_tol Float64 1.0 "offset (zero point) of the cutoff function w.r.t. `rcut`."
@configtag Rmax Int64 1 "maximum magnitude of translation vectors used for periodic boundaty conditions (default: determined by rcut)."
@configtag poscar String "POSCAR" "path to POSCAR file to define base system."
@configtag grid_size Float64 get_rcut(conf) "size of the grid to divide simulation cell."