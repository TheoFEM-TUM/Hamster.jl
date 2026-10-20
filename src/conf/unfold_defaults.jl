# ====================
# Unfold block: band unfolding onto a primitive cell (post-processing of a standard run, switched on by the block)
# ====================
@configtag poscar String "POSCAR_pc" "POSCAR of the primitive cell (Direct coordinates)." "Unfold"
@configtag kpoints String "EIGENVAL_pc" "k-points of the primitive cell, in units of its reciprocal lattice vectors (EIGENVAL or .h5 file with kpoints)." "Unfold"
@configtag method String "exact" "exact (spectral weights of the eigenstates, full or sparse diagonalization) or kpm (kernel polynomial method: spectral function of all states without diagonalization, linear scaling)." "Unfold"
@configtag kpm_moments Int64 1024 "number of Chebyshev moments of the kpm method (energy resolution ≈ π × bandwidth / 2 / kpm_moments)." "Unfold"
@configtag effective_bands Bool true "extract the effective bands (energy and ±1σ width of every primitive band) from the averaged spectral function (needs all states: sp_diag = false or kpm)." "Unfold"
@configtag tol Float64 0.5 "largest distance (Å) between a supercell atom and the nearest periodic image of the primitive site it is mapped to (every site holds at most one atom, assigned closest first)." "Unfold"
@configtag ne Int64 1000 "number of points of the energy grid of the spectral function." "Unfold"
@configtag emin Float64 NaN "lower end of the energy grid (eV). Default: lowest eigenvalue - 5 sigma (exact) or lower spectral bound (kpm)." "Unfold"
@configtag emax Float64 NaN "upper end of the energy grid (eV). Default: highest eigenvalue + 5 sigma (exact) or upper spectral bound (kpm)." "Unfold"
@configtag sigma Float64 0.05 "Gaussian broadening (eV) of the spectral function (exact method)." "Unfold"
@configtag write_weights Bool true "write the results of every structure (exact: eigenvalues and spectral weights; kpm: spectral function and moments)." "Unfold"
@configtag filename String "unfold.h5" "output file of the unfolding." "Unfold"
@configtag displaced_mapping Bool false "map the displaced positions of every MD structure onto the primitive cell instead of the reference positions (finds moving defects; increase tol to about half the shortest distance between sites)." "Unfold"
@configtag band_resolved Bool false "band-resolved unfolding: project onto the band states of the ideal primitive cell (computed with the model of the run) instead of the orbital Bloch states only, which gives the spectral function, energy and width of every primitive band, also where bands overlap (e.g. thermal averages) or with defects." "Unfold"
