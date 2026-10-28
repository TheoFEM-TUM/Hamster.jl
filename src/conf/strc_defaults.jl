# ====================
# Structure defaults
# ====================
@configtag rcut Float64 7.0 "Distance beyond which all interactions are neglected." category="structure" see_also=["rcut_tol", "grid_size"]
@configtag rcut_tol Float64 1.0 "offset (zero point) of the cutoff function w.r.t. `rcut`." category="structure"
@configtag Rmax Int64 1 "maximum magnitude of translation vectors used for periodic boundaty conditions (default: determined by rcut)." category="structure"
@configtag poscar String "POSCAR" "path to POSCAR file to define base system." category="structure"
@configtag grid_size Float64 get_rcut(conf) "size of the grid to divide simulation cell." category="structure" default_doc="same as rcut"