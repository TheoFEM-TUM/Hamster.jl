# ====================
# DOS block: settings of the DOS / PDOS / COHP post-processing (switched on by the `dos` tag in the Options block)
# ====================
@configtag method String "auto" "method: exact (dense diagonalization at all k-points), kpm (kernel polynomial method, sparse, for large supercells) or auto (exact up to `exact_max_orbitals` orbitals, kpm above)." "DOS"
@configtag projection String "element" "projection of PDOS and COHP: element (e.g. Pb, Br), orbital (e.g. Pb-s, Pb-px; spin-summed with SOC) defect (region around the defects found by the mapping onto the primitive cell and element, e.g. defect-Br, shell1-Pb, bulk-Cs) or localized (region around the atoms of localized states, found from the eigenvectors, e.g. localized-Br, shell1-Pb; exact method)." "DOS"
@configtag primitive_poscar String "" "POSCAR of the primitive cell for projection = defect (Direct coordinates). Default: the poscar of the Unfold block." "DOS"
@configtag defect_shells Int64 2 "number of neighbour shells around the defects for projection = defect (shell1, shell2, ...; the remaining atoms are bulk)." "DOS"
@configtag defect_tol Float64 1.0 "largest distance (Å) between an atom (displaced positions of every structure) and the primitive site it is mapped to, for projection = defect; atoms farther away are interstitials." "DOS"
@configtag ne Int64 2000 "number of points of the energy grid." "DOS"
@configtag emin Float64 NaN "lower end of the energy grid (eV). Default: lowest eigenvalue - 5 sigma (exact) or lower spectral bound - 3 kernel widths (kpm)." "DOS"
@configtag emax Float64 NaN "upper end of the energy grid (eV). Default: highest eigenvalue + 5 sigma (exact) or upper spectral bound + 3 kernel widths (kpm)." "DOS"
@configtag sigma Float64 0.05 "broadening width (eV) of the exact method." "DOS"
@configtag broadening String "gaussian" "broadening of the exact method: gaussian or lorentzian." "DOS"
@configtag kpm_moments Int64 1024 "number of Chebyshev moments of the kpm method (energy resolution ≈ π × bandwidth / 2 / kpm_moments)." "DOS"
@configtag kpm_vectors Int64 32 "number of random vectors for the stochastic trace of the kpm method." "DOS"
@configtag exact_max_orbitals Int64 4000 "largest number of orbitals for which method = auto uses exact diagonalization." "DOS"
@configtag nelec Float64 NaN "number of electrons per structure (optional). If set, the Fermi energy, the integrated DOS and the ICOHP are computed." "DOS"
@configtag filename String "dos.h5" "output file of the DOS post-processing." "DOS"

@configtag cohp_pairs Vector{String} String[] "pairs of labels for the COHP, e.g. `cohp_pairs = Pb-Br Cs-Br` (labels as given by `projection`). Default (or all): all pairs coupled by an off-site matrix element, including pairs of the same label on different atoms (e.g. Br-Br)." "DOS"
@configtag localized_threshold Float64 0.1 "projection = localized: states whose participation ratio (number of atoms they live on, relative to a state spread uniformly over the atoms of its elements) is below this value are localized; their participation number gives the number of atoms of the localized region." "DOS"
@configtag localized_emin Float64 NaN "projection = localized: lower end (eV) of the energy window for localized states (default: valence band maximum - 1 eV with nelec, all states otherwise)." "DOS"
@configtag localized_emax Float64 NaN "projection = localized: upper end (eV) of the energy window for localized states (default: conduction band minimum + 1 eV with nelec, all states otherwise)." "DOS"
@configtag defect_separate Bool true "projection = defect or localized: separate regions for every defect (defect atoms, vacancies or localized atoms that are not neighbours), e.g. defect1-Br, defect2-shell1-Pb; false: one region for all defects." "DOS"
