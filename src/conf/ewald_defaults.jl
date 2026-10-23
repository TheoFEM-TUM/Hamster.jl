# ====================
# Ewald defaults
# ====================
"""
    ewald=false

A model to account for long-term electrostatics (Ewald summation) is constructed if a block `Ewald` is present in the config file.
"""
get_ewald(conf::Config)::Bool = haskey(conf.blocks, "Ewald") || (conf("ewald") ≠ "default" ? conf("ewald") : false)
@configtag ewald Bool false "electrostatic on-site model (also switched on by an Ewald block)." getter=:custom default_doc="true if an Ewald block is present, else false" category="model"

@configtag update Bool false "fit the Ewald coupling of each species (starting from charge_scale or init_params)." "Ewald" category="model"
@configtag init_params String "default" "parameter file with the Ewald coupling per species (ewald_<element> lines, e.g. params.dat of a fit), default: charge_scale." "Ewald" category="model" default_doc="none: the couplings start from charge_scale"
@configtag method String "ewald" "method to compute the Ewald sum" "Ewald" category="electrostatics"
@configtag rcut Float64 get_rcut(conf) "cut-off radius for real-space ewald; Wolf and Zahn: range of the electrostatics (alpha = 3 / rcut), in a fit 1.5 × the size of the training cells if neither rcut nor alpha are set." "Ewald" category="electrostatics" default_doc="same as rcut of the Options block (Wolf, Zahn in a fit: 1.5 × the size of the training cells)"
@configtag charge_scale Float64 1.0 "scaling factor for charge values" "Ewald" category="model"
@configtag alpha Float64 0. "Ewald dampening parameter; 0: 4 / rcut for method = ewald, 3 / rcut for wolf and zahn." "Ewald" category="electrostatics" default_doc="0: 4 / rcut for method = ewald, 3 / rcut for wolf and zahn"
@configtag mesh_spacing Float64 0.5 "Grid spacing for reciprocal-space PME." "Ewald" category="electrostatics"
@configtag check_params Bool true "take method, rcut, alpha and qeff from the header of the parameter file if they are not set, stop if they differ (false: use the config)." "Ewald" category="model"