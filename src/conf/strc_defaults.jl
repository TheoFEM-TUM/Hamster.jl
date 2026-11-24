# ====================
# Structure defaults
# ====================
@configtag rcut Float64 7.0 "Distance beyond which all interactions are neglected." category="structure" see_also=["rcut_tol", "grid_size"]
@configtag rcut_tol Float64 1.0 "offset (zero point) of the cutoff function w.r.t. `rcut`." category="structure" see_also=["rcut", "rcut_pair_selection"]
@configtag rcut_pair_selection String "reference" "positions the pair cutoff `rcut` is applied to: reference (reference structure, e.g. the first frame of each system) or frame (current positions; H depends only on the current geometry)." category="structure" values=["reference", "frame"] see_also=["rcut", "rcut_tol", "grid_size"]
@configtag Rmax Int64 1 "maximum magnitude of translation vectors used for periodic boundaty conditions (default: determined by rcut)." category="structure"
@configtag poscar String "POSCAR" "path to POSCAR file to define base system." category="structure"
@configtag grid_size Float64 get_rcut(conf) "size of the grid cells of the neighbor search; must be at least rcut." category="structure" default_doc="same as rcut"