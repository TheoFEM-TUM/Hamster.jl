# ====================
# Unfold block: band unfolding onto a primitive cell (post-processing of a standard run, switched on by the block)
# ====================

"""
    poscar=POSCAR_pc (Unfold)

The `poscar` tag of the `Unfold` block sets the primitive cell onto which the bands are unfolded: a POSCAR (Direct
coordinates; the same primitive cell for all systems) or an `.h5` file in the format of the structures of the
`Supercell` block (`rs_atom` fractional, `lattice`, `atom_types`), read from the group `system_PC` if it exists and
from the group `system` otherwise. With a `Supercell` block, the default is its `xdatcar` file, i.e., the `system_PC`
groups of the structures.
"""
function get_unfold_poscar(conf::Config)::String
    default = haskey(conf, "Supercell") && occursin(".h5", get_xdatcar(conf)) ? get_xdatcar(conf) : "POSCAR_pc"
    return get(conf, "poscar", "Unfold", default)
end

"""
    kpoints=EIGENVAL_pc (Unfold)

The `kpoints` tag of the `Unfold` block sets the k-points of the primitive cell, in units of its reciprocal lattice
vectors: an `EIGENVAL` or VASP `KPOINTS` file (the same k-points for all systems), or an `.h5` file with the dataset
`kpoints`, read from the group `system_PC` if it exists and from the group `system` otherwise (e.g. the eigenvalue file
of the training data, which holds the k-points of the primitive cells). With a `Supercell` block, the default is the
`kpoints` file of the run if it is an `.h5` file.
"""
function get_unfold_kpoints(conf::Config)::String
    default = haskey(conf, "Supercell") && occursin(".h5", get_kpoints_file(conf)) ? get_kpoints_file(conf) : "EIGENVAL_pc"
    return get(conf, "kpoints", "Unfold", default)
end

"""
    method=exact (Unfold)

The `method` tag of the `Unfold` block sets how the spectral function is computed: `exact` (spectral weights of the
eigenstates, full or sparse diagonalization) or `kpm` (kernel polynomial method: spectral function of all states without
diagonalization, linear scaling).
"""
get_unfold_method(conf::Config)::String = get(conf, "method", "Unfold", "exact")

"""
    kpm_moments=1024 (Unfold)

The `kpm_moments` tag sets the number of Chebyshev moments of the kpm method (energy resolution ≈ π × bandwidth / 2 /
kpm_moments).
"""
get_unfold_kpm_moments(conf::Config)::Int64 = get(conf, "kpm_moments", "Unfold", 1024)

"""
    effective_bands=true (Unfold)

The `effective_bands` tag switches on the extraction of the effective bands (energy and ±1σ width of every primitive
band) from the averaged spectral function (needs all states: sp_diag = false or kpm).
"""
get_unfold_effective_bands(conf::Config)::Bool = get(conf, "effective_bands", "Unfold", true)

"""
    tol=0.5 (Unfold)

The `tol` tag sets the largest distance (Å) between a supercell atom and the nearest periodic image of the primitive
site it is mapped to (every site holds at most one atom, assigned closest first).
"""
get_unfold_tol(conf::Config)::Float64 = get(conf, "tol", "Unfold", 0.5)

"""
    ne=1000 (Unfold)

The `ne` tag sets the number of points of the energy grid of the spectral function.
"""
get_unfold_ne(conf::Config)::Int64 = get(conf, "ne", "Unfold", 1000)

"""
    emin=NaN (Unfold)

The `emin` tag sets the lower end of the energy grid (eV). Default: lowest eigenvalue - 5 sigma (exact) or lower
spectral bound (kpm), of every system.
"""
get_unfold_emin(conf::Config)::Float64 = get(conf, "emin", "Unfold", NaN)

"""
    emax=NaN (Unfold)

The `emax` tag sets the upper end of the energy grid (eV). Default: highest eigenvalue + 5 sigma (exact) or upper
spectral bound (kpm), of every system.
"""
get_unfold_emax(conf::Config)::Float64 = get(conf, "emax", "Unfold", NaN)

"""
    sigma=0.05 (Unfold)

The `sigma` tag sets the Gaussian broadening (eV) of the spectral function (exact method).
"""
get_unfold_sigma(conf::Config)::Float64 = get(conf, "sigma", "Unfold", 0.05)

"""
    write_weights=true (Unfold)

The `write_weights` tag switches on writing the results of every structure (exact: eigenvalues and spectral weights;
kpm: spectral function and moments).
"""
get_unfold_write_weights(conf::Config)::Bool = get(conf, "write_weights", "Unfold", true)

"""
    filename=unfold.h5 (Unfold)

The `filename` tag sets the output file of the unfolding.
"""
get_unfold_filename(conf::Config)::String = get(conf, "filename", "Unfold", "unfold.h5")

"""
    displaced_mapping=false (Unfold)

The `displaced_mapping` tag switches on mapping the displaced positions of every MD structure onto the primitive cell
instead of the reference positions (finds moving defects; increase tol to about half the shortest distance between
sites).
"""
get_unfold_displaced_mapping(conf::Config)::Bool = get(conf, "displaced_mapping", "Unfold", false)

"""
    band_resolved=false (Unfold)

The `band_resolved` tag switches on the band-resolved unfolding: the projection onto the band states of the ideal
primitive cell (computed with the model of the run) instead of the orbital Bloch states only, which gives the spectral
function, energy and width of every primitive band, also where bands overlap (e.g. thermal averages) or with defects.
"""
get_unfold_band_resolved(conf::Config)::Bool = get(conf, "band_resolved", "Unfold", false)
