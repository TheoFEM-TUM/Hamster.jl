# ====================
# DOS block: settings of the DOS / PDOS / COHP post-processing (switched on by the `dos` tag in the Options block)
# ====================

"""
    as_string_vector(value)

A config value as `Vector{String}`: `String[]` for `"default"`, a one-element vector for a single word.
"""
as_string_vector(value)::Vector{String} = value == "default" ? String[] : value isa AbstractVector ? String.(string.(value)) : [String(string(value))]

"""
**dos**=none

Post-processing of a standard run: any combination of `dos` (total DOS), `pdos` (projected DOS) and `cohp` (crystal
orbital Hamilton populations), or `all` (dos, pdos and cohp), e.g. `dos = pdos` or `dos = dos pdos`. The standard run is
unchanged; the results are additionally written to `dos.h5`. Further settings in the `DOS` block.
"""
get_dos(conf::Config)::Vector{String} = as_string_vector(conf("dos"))

"""
    method=auto (DOS)

`exact` (dense diagonalization at all k-points), `kpm` (kernel polynomial method, sparse, for large supercells) or `auto`
(exact up to `exact_max_orbitals` orbitals, kpm above).
"""
get_dos_method(conf::Config)::String = get(conf, "method", "DOS", "auto")

"""
    projection=element (DOS)

Projection of the PDOS and COHP: `element` (e.g. Pb, Br) or `orbital` (e.g. Pb-s, Pb-px; spin-summed with SOC).
"""
get_dos_projection(conf::Config)::String = get(conf, "projection", "DOS", "element")

"""
    cohp_pairs=all (DOS)

Pairs of labels for the COHP, e.g. `cohp_pairs = Pb-Br Cs-Br` (labels as given by `projection`). Default (or `all`): all
pairs coupled by an off-site matrix element, including pairs of the same label on different atoms (e.g. Br-Br).
"""
get_dos_cohp_pairs(conf::Config)::Vector{String} = as_string_vector(conf("cohp_pairs", "DOS"))

"""
    ne=2000 (DOS)

Number of points of the energy grid.
"""
get_dos_ne(conf::Config)::Int64 = get(conf, "ne", "DOS", 2000)

"""
    emin=auto (DOS)

Lower end of the energy grid (eV). Default: lowest eigenvalue - 5 sigma (exact) or lower spectral bound (kpm).
"""
get_dos_emin(conf::Config)::Float64 = get(conf, "emin", "DOS", NaN)

"""
    emax=auto (DOS)

Upper end of the energy grid (eV). Default: highest eigenvalue + 5 sigma (exact) or upper spectral bound (kpm).
"""
get_dos_emax(conf::Config)::Float64 = get(conf, "emax", "DOS", NaN)

"""
    sigma=0.05 (DOS)

Broadening width (eV) of the exact method.
"""
get_dos_sigma(conf::Config)::Float64 = get(conf, "sigma", "DOS", 0.05)

"""
    broadening=gaussian (DOS)

Broadening of the exact method: `gaussian` or `lorentzian`.
"""
get_dos_broadening(conf::Config)::String = get(conf, "broadening", "DOS", "gaussian")

"""
    kpm_moments=1024 (DOS)

Number of Chebyshev moments of the kpm method (energy resolution ≈ π × bandwidth / 2 / kpm_moments).
"""
get_dos_kpm_moments(conf::Config)::Int64 = get(conf, "kpm_moments", "DOS", 1024)

"""
    kpm_vectors=32 (DOS)

Number of random vectors for the stochastic trace of the kpm method.
"""
get_dos_kpm_vectors(conf::Config)::Int64 = get(conf, "kpm_vectors", "DOS", 32)

"""
    exact_max_orbitals=4000 (DOS)

Largest number of orbitals for which `method = auto` uses exact diagonalization.
"""
get_dos_exact_max_orbitals(conf::Config)::Int64 = get(conf, "exact_max_orbitals", "DOS", 4000)

"""
    nelec=not set (DOS)

Number of electrons per structure (optional). If set, the Fermi energy, the integrated DOS and the ICOHP are computed.
"""
get_dos_nelec(conf::Config)::Float64 = get(conf, "nelec", "DOS", NaN)

"""
    filename=dos.h5 (DOS)

Output file of the DOS post-processing.
"""
get_dos_filename(conf::Config)::String = get(conf, "filename", "DOS", "dos.h5")
