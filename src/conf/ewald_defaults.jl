# ====================
# Ewald defaults
# ====================
"""
    ewald=false

A model to account for long-term electrostatics (Ewald summation) is constructed if a block `Ewald` is present in the config file.
"""
get_ewald(conf::Config)::Bool = haskey(conf.blocks, "Ewald") || (conf("ewald") ≠ "default" ? conf("ewald") : false)
@configtag ewald Bool false "electrostatic on-site model (also switched on by an Ewald block)." getter=:custom default_doc="true if an Ewald block is present, else false" category="model"

@configtag update Bool false "fit the Ewald coupling of each species and orbital type s, p, d (starting from charge_scale, charge_scale_<l> or init_params)." "Ewald" category="model"
@configtag update_polarizability Bool get_ewald_update(conf) "fit the polarizabilities of the polarizable species (atom tag polarizability)." "Ewald" category="model" default_doc="same as update" see_also=["polarizability"]
@configtag init_params String "default" "parameter file with the Ewald couplings (ewald_<element> or ewald_<element>_<l> lines, <l> = s, p or d) and polarizabilities (polarizability_<element> lines), e.g. params.dat of a fit; default: charge_scale and polarizability of the config." "Ewald" category="model" default_doc="none: the couplings start from charge_scale, the polarizabilities from polarizability"
@configtag method String "ewald" "method of the electrostatic sum: ewald (Ewald/PME), wolf or zahn (case-insensitive)." "Ewald" category="electrostatics" values=["ewald", "wolf", "zahn"]
@configtag rcut Float64 get_rcut(conf) "cut-off radius for real-space ewald; Wolf and Zahn: range of the electrostatics (alpha = 3 / rcut), in a fit 1.5 × the size of the training cells if neither rcut nor alpha are set." "Ewald" category="electrostatics" default_doc="same as rcut of the Options block (Wolf, Zahn in a fit: 1.5 × the size of the training cells)"
@configtag charge_scale Float64 1.0 "scaling factor for charge values" "Ewald" category="model"
@configtag alpha Float64 0. "Ewald dampening parameter; 0: 4 / rcut for method = ewald, 3 / rcut for wolf and zahn." "Ewald" category="electrostatics" default_doc="0: 4 / rcut for method = ewald, 3 / rcut for wolf and zahn"
@configtag mesh_spacing Float64 0.5 "Grid spacing for reciprocal-space PME." "Ewald" category="electrostatics"
@configtag slab_axis Int64 0 "lattice vector (1, 2 or 3) across the vacuum of slab structures; 0: periodic (no slab). With method = wolf or zahn, the plane-averaged long-range part of the electrostatics is added to the potentials (aligned dipoles of successive layers, which the truncated sum misses); with method = ewald, the dipole correction." "Ewald" category="electrostatics"
@configtag slab_scale Float64 1.0 "factor of the long-range slab term (slab_axis; plane-averaged potential for wolf and zahn, dipole correction for ewald), e.g. for the screening of the macroscopic field (≈ 1/ε∞), which can differ from the couplings of the short-range part." "Ewald" category="electrostatics"
@configtag cluster Bool false "isolated cluster (e.g. a quantum dot) in a vacuum box: the dipole correction of the full Ewald sum in all directions (vacuum instead of tinfoil boundary conditions; method = ewald, for boxes close to cubic). The net charge of global charge_compensation is a uniform background on the atoms of the cluster." "Ewald" category="electrostatics" see_also=["slab_axis", "charge_compensation"]
"""
    charge_compensation = surface if slab_axis > 0, else global

Where the net effective charge of a structure is compensated in the electrostatics (see the Ewald documentation).
"""
function get_ewald_charge_compensation(conf::Config)::String
    mode = conf("charge_compensation", "Ewald")
    mode == "default" && return get_ewald_slab_axis(conf) > 0 ? "surface" : "global"
    return lowercase(string(mode))
end
@configtag charge_compensation String "default" "where the net effective charge Σq of a structure is compensated: global (uniform background over all atoms, for the slab terms; with cluster, in the whole sum), surface (on the outermost layers of a slab along slab_axis, split so that no average field remains inside), local (on the atoms with missing neighbors of opposite charge, e.g. next to a vacancy, in proportion to the missing neighbors). surface charges enter only the slab terms, local charges also the short-range sum." "Ewald" category="electrostatics" getter=:custom default_doc="surface if slab_axis > 0, else global" see_also=["slab_axis", "cluster"]
"""
    centering = structure if slab_axis > 0 or cluster, else global

Which species means are subtracted from the Ewald potentials (see the Ewald documentation).
"""
function get_ewald_centering(conf::Config)::String
    mode = conf("centering", "Ewald")
    mode == "default" && return get_ewald_slab_axis(conf) > 0 || get_ewald_cluster(conf) ? "structure" : "global"
    return lowercase(string(mode))
end
@configtag centering String "default" "species means subtracted from the potentials: global (the mean of each species over all structures of a run) or structure (the means of each structure, so that the onsite corrections of a structure do not depend on the other structures and a potential offset that differs between structures, e.g. of the slab or cluster terms, drops out)." "Ewald" category="electrostatics" getter=:custom default_doc="structure if slab_axis > 0 or cluster, else global" see_also=["slab_axis", "cluster"]
@configtag site_rcut Float64 2.0 "search radius (Å) for the bonded neighbour of the site-dependent charges qeff_<element> of the atom blocks; at most rcut of the Options block." "Ewald" category="electrostatics" see_also=["qeff"]
@configtag check_params Bool true "take method, rcut, alpha and qeff from the header of the parameter file if they are not set, stop if they differ (false: use the config)." "Ewald" category="model"