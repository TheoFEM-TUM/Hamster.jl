# ====================
# DOS block: settings of the DOS / PDOS / COHP post-processing (switched on by the `dos` tag in the Options block)
# ====================
@configtag method String "auto" "method: exact (dense diagonalization at all k-points), kpm (kernel polynomial method, sparse, for large supercells) or auto (exact up to `exact_max_orbitals` orbitals, kpm above)." "DOS" category="method"
@configtag projection String "element" "projection of PDOS and COHP: element (e.g. Pb, Br) or orbital (e.g. Pb-s, Pb-px; spin-summed with SOC)." "DOS" category="quantities"
@configtag ne Int64 2000 "number of points of the energy grid." "DOS" category="energy grid and broadening"
@configtag emin Float64 NaN "lower end of the energy grid (eV). Default: lowest eigenvalue - 5 sigma (exact) or lower spectral bound (kpm)." "DOS" category="energy grid and broadening" default_doc="automatic (see description)"
@configtag emax Float64 NaN "upper end of the energy grid (eV). Default: highest eigenvalue + 5 sigma (exact) or upper spectral bound (kpm)." "DOS" category="energy grid and broadening" default_doc="automatic (see description)"
@configtag sigma Float64 0.05 "broadening width (eV) of the exact method." "DOS" category="energy grid and broadening"
@configtag broadening String "gaussian" "broadening of the exact method: gaussian or lorentzian." "DOS" category="energy grid and broadening"
@configtag kpm_moments Int64 1024 "number of Chebyshev moments of the kpm method (energy resolution ≈ π × bandwidth / 2 / kpm_moments)." "DOS" category="method"
@configtag kpm_vectors Int64 32 "number of random vectors for the stochastic trace of the kpm method." "DOS" category="method"
@configtag exact_max_orbitals Int64 4000 "largest number of orbitals for which method = auto uses exact diagonalization." "DOS" category="method"
@configtag nelec Float64 NaN "number of electrons per structure (optional). If set, the Fermi energy, the integrated DOS and the ICOHP are computed." "DOS" category="electrons" default_doc="not set"
@configtag filename String "dos.h5" "output file of the DOS post-processing." "DOS" category="output"

@configtag cohp_pairs Vector{String} String[] "pairs of labels for the COHP, e.g. `cohp_pairs = Pb-Br Cs-Br` (labels as given by `projection`). Default (or all): all pairs coupled by an off-site matrix element, including pairs of the same label on different atoms (e.g. Br-Br)." "DOS" category="quantities" default_doc="all coupled pairs"
