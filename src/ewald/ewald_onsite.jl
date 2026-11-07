"""
    EwaldOnsites{P,T,N}

Container for electrostatic onsite corrections computed from Ewald-type
methods (Ewald, Wolf, Zahn) over a set of structures.

This struct stores per-atom electrostatic potentials (mean-centered
per species) together with the metadata required to map them onto TB
onsite parameters.

# Fields
- `params::Vector{Float64}`:
    The couplings γ (one per entry of `coupling_labels`), followed by the polarizabilities (Å³, one per entry of
    `pol_labels`).

- `coupling_labels::Vector{Tuple{UInt8, Int64}}`:
    Species and orbital angular momentum `l` of each coupling (`coupling = orbital`); `l = -1` for one coupling of all
    orbitals of the species (`coupling = species`), species 0 for one coupling of all species (`coupling = global`).

- `species::Vector{UInt8}`:
    Species with orbitals in the structures.

- `pol_labels::Vector{UInt8}`:
    Polarizable species (atom tag `polarizability`).

- `potentials::Vector{P}`:
    Electrostatic potentials of the effective charges for each structure. Each entry is a vector of
    site potentials (one per atom), typically computed via an Ewald-like method
    and centered per species.

- `dipole_potentials::Vector{Matrix{Float64}}`:
    Potentials (atoms × polarizable species) of the first-order dipoles induced on the atoms of each polarizable species
    with polarizability 1 Å³ (`induced_dipole_potentials`), centered per species like `potentials`.

- `types_per_strc::Vector{T}`:
    Atomic species (as `UInt8`) for each atom in each structure.

- `norb_per_strc::Vector{N}`:
    Number of orbitals per structure (used to map site potentials to Hamiltonian
    orbital indices).

- `coupling_per_strc::Vector{Vector{Int64}}`:
    Index of the coupling (in `coupling_labels`) of every orbital of each structure (without spin).

- `Rs_info::Matrix{Int64}`:
    Auxiliary information about lattice translations per structure:
    - row 1: number of R-vectors
    - row 2: index of the reference cell (R = 0)

- `update::Bool`:
    Flag indicating whether the couplings are updated during optimization/training.

- `update_polarizability::Bool`:
    Flag indicating whether the polarizabilities are updated during optimization/training.

# Description
The `EwaldOnsites` object encapsulates the electrostatic contribution to
tight-binding onsite energies, computed from effective charges via Ewald
or related methods. Potentials are typically centered per species across
all structures to remove arbitrary reference offsets.

This allows a mapping of the form:

    Δε_o = -γ_{s(i), l(o)} ⋅ (φᵢ + Σ_t p_t Φ_{i,t})

for the orbital `o` with angular momentum `l(o)` of atom `i` (species `s(i)`), where `φᵢ` are the stored potentials,
`Φ` the dipole potentials, `γ` the couplings and `p` the polarizabilities in `params`.

# Constructor
    EwaldOnsites(strcs, bases, comm, conf; kwargs...)

Builds the container by:

1. Computing electrostatic potentials for each structure
2. Gathering unique species across MPI ranks
3. Computing the induced-dipole potentials of the polarizable species
4. Initializing the couplings per species and orbital angular momentum, and the polarizabilities
5. Computing global per-species mean potentials (MPI-reduced)
6. Subtracting these means from all site potentials (`centering = global`), or the species means of each structure
   (`centering = structure`)

# Notes
- Potentials are only defined up to a constant; per-species centering ensures
  consistency with onsite fitting.
- The choice of electrostatic method (Ewald, Wolf, Zahn) affects the physical
  interpretation of the stored potentials.
"""
mutable struct EwaldOnsites{P, T, N}
    params :: Vector{Float64}
    coupling_labels :: Vector{Tuple{UInt8, Int64}}
    species :: Vector{UInt8}
    pol_labels :: Vector{UInt8}
    potentials :: Vector{P}
    dipole_potentials :: Vector{Matrix{Float64}}
    types_per_strc :: Vector{T}
    norb_per_strc :: Vector{N}
    coupling_per_strc :: Vector{Vector{Int64}}
    Rs_info :: Matrix{Int64}
    update :: Bool
    update_polarizability :: Bool
end

function EwaldOnsites(strcs::Vector{Structure}, bases::Vector{<:Basis}, comm, conf=get_empty_config(); 
        rank=0,
        method          = get_ewald_method(conf),
        rcut            = get_ewald_rcut(conf),
        alpha           = get_ewald_alpha(conf),
        mesh_spacing    = get_ewald_mesh_spacing(conf),
        slab_axis       = get_ewald_slab_axis(conf),
        slab_scale      = get_ewald_slab_scale(conf),
        cluster         = get_ewald_cluster(conf),
        charge_compensation = get_ewald_charge_compensation(conf),
        centering       = conf("centering", "Ewald") == "default" ? (slab_axis > 0 || cluster ? "structure" : "global") : get_ewald_centering(conf),
        update          = get_ewald_update(conf),
        update_polarizability = get_ewald_update_polarizability(conf),
        coupling        = get_ewald_coupling(conf),
        label           = "",
        write_summary   = true)

    method = ewald_method(method)
    α = ewald_alpha(method, rcut, alpha)
    coupling = lowercase(coupling)
    pol_labels = polarizable_species(conf)
    check_ewald_options(conf; method=method, slab_axis=slab_axis, cluster=cluster, coupling=coupling, polarizable=!isempty(pol_labels))
    site_counts = Dict{Tuple{String, Float64}, Int64}()
    settings = (method=method, rcut=rcut, alpha=α, mesh_spacing=mesh_spacing, slab_axis=slab_axis, slab_scale=slab_scale,
                cluster=cluster, charge_compensation=charge_compensation)
    sites = map(enumerate(strcs)) do (n, strc)
        site_potentials_of_structure(strc, bases[n], conf, n, pol_labels, site_counts; coupling=coupling, settings...)
    end
    types_per_strc = [s.types for s in sites]; norb_per_strc = [s.norb for s in sites]
    potentials = [s.φ for s in sites]; dipole_potentials = [s.Φ for s in sites]
    Rs_info = reduce(hcat, [[size(strc.Rs, 2), findR0(strc.Rs)] for strc in strcs], init=zeros(Int64, 2, 0))
    grid_rebuilt = any(s -> s.grid_rebuilt, sites)

    coupling_labels, coupling_per_strc = ewald_coupling_labels(sites, coupling, comm, rank=rank)
    species = gather_unique(reduce(vcat, types_per_strc, init=UInt8[]), comm, rank=rank)
    params = init_ewald_params(coupling_labels, pol_labels, conf, species=species)
    species_mean, structure_means = center_potentials!(potentials, dipole_potentials, types_per_strc, species,
                                                        params[length(coupling_labels)+1:end], centering, comm)

    model = EwaldOnsites(params, coupling_labels, species, pol_labels, potentials, dipole_potentials, types_per_strc, norb_per_strc,
                         coupling_per_strc, Rs_info, update, update_polarizability)
    has_sites = any(!isempty(get_site_qeff(conf, type)) for type in keys(conf.blocks))
    write_summary && write_model_summary(model, comm, conf, rank=rank, label=label, species=species, species_mean=species_mean,
                                         structure_means=structure_means, method=method, rcut=rcut, alpha=α, alpha_auto=alpha == 0,
                                         grid_rebuilt=grid_rebuilt && method == "ewald", slab_axis=slab_axis, slab_scale=slab_scale,
                                         cluster=cluster, charge_compensation=charge_compensation, centering=centering,
                                         site_counts=has_sites ? site_counts : nothing)
    return model
end

"""
    check_ewald_options(conf; method, slab_axis, cluster, coupling, polarizable)

Stop with an error for settings of the Ewald model that exclude each other: `cluster` with `slab_axis` or without
`method = ewald`; an unknown `coupling`; `charge_scale` in an atom block with `coupling = global`; `charge_scale_<l>` without
`coupling = orbital`; polarizable species (`polarizable`) without `method = wolf` or `zahn`, or with slabs and clusters.
"""
function check_ewald_options(conf; method, slab_axis, cluster, coupling, polarizable)
    if cluster
        slab_axis == 0 || error("cluster and slab_axis = $slab_axis exclude each other: a cluster has vacuum in all directions, a slab along one.")
        method == "ewald" || error("cluster = true needs method = ewald (the full Ewald sum with the dipole correction); " *
            "Wolf and Zahn with rcut larger than the cluster already sum an isolated cluster.")
    end
    coupling ∈ ("global", "species", "orbital") || error("coupling = $coupling in the Ewald block: use global, species or orbital.")
    for type in keys(conf.blocks), key in ("charge_scale_s", "charge_scale_p", "charge_scale_d", "charge_scale")
        (type == "Ewald" || conf(key, type) == "default") && continue
        key == "charge_scale" && coupling == "global" && error("charge_scale in the block of $type: coupling = global uses only the charge_scale of the Ewald block.")
        key ≠ "charge_scale" && coupling ≠ "orbital" && error("$key in the block of $type needs coupling = orbital in the Ewald block.")
    end
    if polarizable
        method ∈ ("wolf", "zahn") || error("Polarizabilities (induced dipoles) need method = wolf or zahn in the Ewald block, not $method.")
        (slab_axis > 0 || cluster) && error("Polarizabilities (induced dipoles) are not available for slabs and clusters (slab_axis, cluster).")
    end
    return nothing
end

"""
    site_potentials_of_structure(strc, basis, conf, n, pol_labels, site_counts; coupling, method, rcut, alpha, mesh_spacing,
                                 slab_axis, slab_scale, cluster, charge_compensation)

Electrostatic data of structure `n` for the atoms with orbitals: species `types`, numbers of orbitals `norb`, orbital
angular momenta `l` (-1 unless `coupling = orbital`), potentials `φ` of the effective charges (with the slab or cluster
terms) and potentials `Φ` of the induced dipoles of the species `pol_labels` (`induced_dipole_potentials`); `grid_rebuilt`
if the real-space grid of the structure was too small for `rcut`. Adds the charges per site to `site_counts`.
"""
function site_potentials_of_structure(strc, basis, conf, n, pol_labels, site_counts; coupling, method, rcut, alpha, mesh_spacing,
                                      slab_axis, slab_scale, cluster, charge_compensation)
    pos = get_ion_positions(strc.ions, apply_distortion=true)
    ion_types = get_ion_types(strc.ions)
    charges, sites = get_site_charges(strc, conf, positions=pos)
    slab_axis > 0 || cluster || charge_compensation ≠ "global" || check_neutrality(charges, sites, strc.system, n)
    for site in zip(sites, charges); site_counts[site] = get(site_counts, site, 0) + 1; end
    # from the reference positions: the same compensation for all distortions of a structure
    slab_charges = charges .+ compensation_charges(charge_compensation, get_ion_positions(strc.ions, apply_distortion=false),
                                                   ion_types, charges, strc.lattice; slab_axis=slab_axis)
    charge_compensation == "local" && (charges = slab_charges)
    cluster && charge_compensation == "global" && (charges = charges .- sum(charges) / length(charges))

    reuse_grid = rcut ≤ min(strc.point_grid.grid_size, get_rcut(conf))
    grid = method == "ewald" && reuse_grid ? (Rs=strc.Rs, point_grid=strc.point_grid) : (Rs=nothing, point_grid=nothing)
    φ = ewald_sum(pos, charges, strc.lattice; rcut=rcut, mesh_spacing=mesh_spacing, method=method, alpha=alpha, grid...).potentials
    slab_axis > 0 && (φ = φ .+ slab_scale .* slab_potentials(pos, slab_charges, strc.lattice, slab_axis; method=method, alpha=alpha))
    cluster && (φ = φ .+ cluster_potentials(pos, charges, strc.lattice; method=method))
    groups = [something(findfirst(==(t), pol_labels), 0) for t in ion_types]
    Φ = isempty(pol_labels) ? zeros(length(ion_types), 0) :
        induced_dipole_potentials(pos, charges, strc.lattice, groups, length(pol_labels); rcut=rcut, alpha=alpha, method=method)

    types_with_orbs = get_ion_types(strc.ions, conf, uniq=true, withorbitals=true)
    ions = filter(i -> ion_types[i] ∈ types_with_orbs, eachindex(ion_types))
    l = [coupling == "orbital" ? orb.type.l : -1 for i in ions for orb in basis.orbitals[i]]
    return (types=ion_types[ions], norb=size(basis)[ions], l=l, φ=φ[ions], Φ=Φ[ions, :], grid_rebuilt=!reuse_grid)
end

"""
    gather_unique(values, comm; rank=0)

The sorted unique `values` of all ranks of `comm`, on all ranks.
"""
function gather_unique(values, comm; rank=0)
    all_values = MPI.gather(unique(values), comm, root=0)
    return MPI.bcast(rank == 0 ? sort(unique(reduce(vcat, all_values))) : eltype(values)[], 0, comm)
end

"""
    ewald_coupling_labels(sites, coupling, comm; rank=0)

Couplings of the Ewald model over all structures (`sites` from `site_potentials_of_structure`) and ranks: the labels
`(species, l)` (`l = -1` for all orbitals of a species, species 0 for all species with `coupling = global`) and, per
structure, the coupling index of every orbital.
"""
function ewald_coupling_labels(sites, coupling, comm; rank=0)
    key(t) = coupling == "global" ? 0x00 : t
    orbital_labels(s) = [(key(t), l) for (t, l) in zip(reduce(vcat, [fill(t, k) for (t, k) in zip(s.types, s.norb)], init=UInt8[]), s.l)]
    labels = gather_unique(reduce(vcat, orbital_labels.(sites), init=Tuple{UInt8, Int64}[]), comm, rank=rank)
    return labels, [[findfirst(==(label), labels) for label in orbital_labels(s)] for s in sites]
end

"""
    center_potentials!(potentials, dipole_potentials, types_per_strc, species, polarizabilities, centering, comm)

Subtract the species means from the charge potentials and from every column of the dipole potentials: the means over
all structures and ranks (`centering = global`) or of each structure (`structure`). Returns the species means of the
total potential `φ + Φ p` over all structures and the per-structure species means `(species, mean)` of the total potential
before centering (for the model summary).
"""
function center_potentials!(potentials, dipole_potentials, types_per_strc, species, polarizabilities, centering, comm)
    centering ∈ ("global", "structure") || error("centering = $centering: unknown, use global or structure.")
    index = Dict(t => i for (i, t) in enumerate(species))
    columns = [potentials, [[Φ[:, t] for Φ in dipole_potentials] for t in eachindex(polarizabilities)]...]
    column_means = map(columns) do col
        sums = zeros(length(species)); counts = zeros(Int64, length(species))
        for n in eachindex(col), (v, type) in zip(col[n], types_per_strc[n])
            sums[index[type]] += v; counts[index[type]] += 1
        end
        MPI.Allreduce(sums, +, comm) ./ max.(MPI.Allreduce(counts, +, comm), 1)
    end
    species_mean = column_means[1] .+ sum((column_means[1+t] .* p for (t, p) in enumerate(polarizabilities)), init=zeros(length(species)))
    structure_means = Tuple{UInt8, Float64}[]
    for n in eachindex(potentials)
        types = types_per_strc[n]
        total = potentials[n] .+ dipole_potentials[n] * polarizabilities
        append!(structure_means, [(type, mean(total[types .== type])) for type in unique(types)])
        for (k, v) in enumerate([potentials[n], eachcol(dipole_potentials[n])...])
            means = Dict(type => mean(v[types .== type]) for type in unique(types))
            v .-= [centering == "structure" ? means[t] : column_means[k][index[t]] for t in types]
        end
    end
    return species_mean, structure_means
end

"""
    polarizable_species(conf; initas=get_ewald_init_params(conf))

Polarizable species (atomic numbers, sorted): the species with the tag `polarizability` in their atom block and, with a
parameter file `initas`, the species of its `polarizability_<element>` lines.
"""
function polarizable_species(conf; initas=get_ewald_init_params(conf))
    types = [type for type in keys(conf.blocks) if haskey(elements, Symbol(type)) && conf("polarizability", type) ≠ "default"]
    if initas ≠ "default"
        append!(types, read_params(initas)[8])
    end
    return sort(unique(element_to_number.(types)))
end

"""
    get_site_charges(strc, conf; positions=get_ion_positions(strc.ions, apply_distortion=true), site_rcut=get_ewald_site_rcut(conf))

Effective charges of the atoms of `strc` and their site labels. An atom gets the `qeff` of its species, unless the atom
block of the species has site-dependent charges `qeff_<element>`: then the nearest neighbour among these elements within
`site_rcut` decides, e.g. with `qeff_N` and `qeff_C` in the block of H, an H atom bonded to N gets `qeff_N` and the site
label `H(N)`. Atoms without such a neighbour keep `qeff` and the label of the species, which usually breaks the charge
neutrality of the cell (see `check_neutrality`). The neighbours are searched on a grid with cell size `site_rcut`, so
the cost is linear in the number of atoms.
"""
function get_site_charges(strc::Structure, conf=get_empty_config();
                          positions=get_ion_positions(strc.ions, apply_distortion=true),
                          site_rcut=get_ewald_site_rcut(conf))
    ion_types = get_ion_types(strc.ions)
    species = string.(number_to_element.(ion_types))
    charges = [get_qeff(conf, s) for s in species]
    sites = copy(species)
    rules = Dict(s => Dict(element_to_number(e) => q for (e, q) in get_site_qeff(conf, s)) for s in unique(species))
    any(!isempty, values(rules)) || return charges, sites
    site_rcut ≤ get_rcut(conf) || error("site_rcut = $site_rcut of the Ewald block is larger than rcut = $(get_rcut(conf)).")

    neighbour_types = union((keys(r) for r in values(rules))...)
    candidates = findall(t -> t ∈ neighbour_types, ion_types)
    isempty(candidates) && return charges, sites
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    grid = get_grid_dict(reduce(hcat, positions[candidates]), Ts, site_rcut)
    for (i, s) in enumerate(species)
        isempty(rules[s]) && continue
        dmin, jmin = Inf, 0
        for grid_point in nn_grid_points(get_grid_point(positions[i], site_rcut), grid), (c, R) in grid[grid_point]
            j = candidates[c]
            (j == i || !haskey(rules[s], ion_types[j])) && continue
            d = norm(positions[i] - apply_pbc(positions[j], Ts[:, R]))
            d < dmin && ((dmin, jmin) = (d, j))
        end
        if dmin ≤ site_rcut
            charges[i] = rules[s][ion_types[jmin]]
            sites[i] = "$s($(species[jmin]))"
        end
    end
    return charges, sites
end

"""
    check_neutrality(charges, sites, system="", index=1; tol=1e-8)

Stop with an error if the effective charges of a structure do not add up to zero, listing the charges of the sites.
With site-dependent charges (`qeff_<element>`), a net charge also indicates an atom without the expected neighbour,
e.g. an H atom that is not bonded to C or N.
"""
function check_neutrality(charges, sites, system="", index=1; tol=1e-8)
    net = sum(charges)
    abs(net) < tol && return nothing
    counts = Dict{Tuple{String, Float64}, Int64}()
    for site in zip(sites, charges); counts[site] = get(counts, site, 0) + 1; end
    listing = join(["$(n) × $(site) ($(q))" for ((site, q), n) in sort(collect(counts))], ", ")
    error("The effective charges of structure $index$(isempty(system) ? "" : " ($system)") are not neutral: net charge " *
          "$(round(net, sigdigits=6)) e from $listing. Check qeff and qeff_<element> in the atom blocks; with " *
          "site-dependent charges, atoms without a neighbour of the given elements within site_rcut keep qeff. A net " *
          "charge of slabs, clusters or defects can be compensated (slab_axis, cluster or charge_compensation in the Ewald block).")
end

"""
    ewald_alpha(method, rcut, alpha)

Damping α of the Ewald sum: `alpha` if set (≠ 0), else 4 / rcut for `ewald`, where α only splits the sum into a real-
and a reciprocal-space part, and 3 / rcut for `wolf` and `zahn`, where α sets the screening length (erfc(3) ≈ 2⋅10⁻⁵).
"""
ewald_alpha(method, rcut, alpha) = alpha ≠ 0 ? Float64(alpha) : (lowercase(method) ∈ ("wolf", "zahn") ? 3.0 : 4.0) / rcut

"""
    ewald_method(method)

The Ewald method `method` (tag `method` of the `Ewald` block) in lower case: `ewald`, `wolf` or `zahn`. Other values
raise an error.
"""
function ewald_method(method)
    m = lowercase(string(method))
    m ∈ ("ewald", "wolf", "zahn") || error("Unknown Ewald method '$method'. Valid options are ewald, wolf and zahn.")
    return m
end

"""
    ewald_settings(conf)

Settings that define the Ewald potentials: method, rcut, the damping α actually used and the charges `qeff`, written
to the header of the parameter file of a fit (see `apply_ewald_settings!`). Site-dependent charges `qeff_<element>` of
the species `type` are listed as `<type>_<element>`.
"""
function ewald_settings(conf)
    method, rcut = ewald_method(get_ewald_method(conf)), get_ewald_rcut(conf)
    qeff = [(type, get_qeff(conf, type)) for (type, block) in conf.blocks if haskey(block, "qeff")]
    for type in keys(conf.blocks), (element, q) in get_site_qeff(conf, type)
        push!(qeff, ("$(type)_$(element)", q))
    end
    sort!(qeff)
    return (method=method, rcut=rcut, alpha=ewald_alpha(method, rcut, get_ewald_alpha(conf)), qeff=qeff)
end

"""
    cell_size(lattice)

Largest perpendicular width of the cell spanned by the columns of `lattice`, i.e., the longest wavelength of a
periodic potential in the cell (the smallest reciprocal lattice vector is 2π / cell_size).
"""
cell_size(lattice) = maximum(abs(det(lattice)) / norm(cross(lattice[:, j], lattice[:, k])) for (j, k) in ((2, 3), (3, 1), (1, 2)))

"""
    apply_ewald_settings!(conf; strcs=Structure[], comm=nothing, training=false, rank=0, verbosity=get_verbosity(conf))

Ewald settings of a model, before it is built:

1. Use a fitted model with the settings it was fitted with: the settings in the header of the parameter file
   (`init_params` of the `Ewald` block, else `init_params`) that are not set in the config are taken from the file,
   e.g., for larger cells. Settings set in the config that differ from the file raise an error, unless
   `check_params = false` in the `Ewald` block. For `method = ewald`, rcut and α only affect the numerics and are not
   taken from the file.
2. Wolf and Zahn, where α = 3 / rcut (`ewald_alpha`) unless `alpha` is set: in a fit (`training`) without `rcut` and
   `alpha`, rcut = 1.5 L with the size L of the training cells (`cell_size`, largest over `strcs` and the ranks of `comm`),
   i.e. α = 2 / L, so that the potential of the longest wavelength in the cells is kept to ≥ 90 % (1 - exp(-k²/4α²) at
   k = 2π/L). With only `alpha` given, rcut = 3 / α. Both are written to the header of the parameter file at the end of
   the fit. Without a fit, rcut or alpha have to come from the file or the config.
"""
function apply_ewald_settings!(conf; strcs=Structure[], comm=nothing, training=false, rank=0, verbosity=get_verbosity(conf))
    adopt_ewald_settings!(conf, rank=rank, verbosity=verbosity)
    ewald_method(get_ewald_method(conf)) ∈ ("wolf", "zahn") || return nothing
    set_rcut, set_alpha = conf("rcut", "Ewald") ≠ "default", conf("alpha", "Ewald") ≠ "default"
    (set_alpha && set_rcut) && return nothing
    if set_alpha
        set_value!(conf, "rcut", "Ewald", 3 / get_ewald_alpha(conf))
        return nothing
    end
    set_rcut && return nothing
    if !training
        error("Ewald method $(get_ewald_method(conf)) without rcut and alpha: they are set from the size of the training " *
              "cells in a fit and written to its params.dat. Use the params.dat of a fit (init_params) or set rcut or alpha " *
              "in the Ewald block.")
    end
    L = isempty(strcs) ? 0.0 : maximum(cell_size(strc.lattice) for strc in strcs)
    isnothing(comm) || (L = MPI.Allreduce(L, max, comm))
    set_value!(conf, "rcut", "Ewald", 1.5 * L)
    if rank == 0 && verbosity > 0
        @printf("Wolf range from the training cells (L = %.3f Å): rcut = 1.5 L = %.3f Å, alpha = 3 / rcut = %.4f 1/Å\n", L, 1.5 * L, 2 / L)
    end
    return nothing
end

"""
    adopt_ewald_settings!(conf; rank=0, verbosity=get_verbosity(conf))

Take the Ewald settings (method, and rcut and α for Wolf and Zahn, charges `qeff`) from the header of the parameter file
of the model (`init_params` of the `Ewald` block, else `init_params`) where they are not set in `conf`, and stop if a
setting in `conf` differs from the file (unless `check_params = false`). See `apply_ewald_settings!`.
"""
function adopt_ewald_settings!(conf; rank=0, verbosity=get_verbosity(conf))
    get_ewald_check_params(conf) || return nothing
    file = get_ewald_init_params(conf) ≠ "default" ? get_ewald_init_params(conf) : string(get_init_params(conf))
    isfile(file) || return nothing
    header = read_params(file)[5]
    haskey(header, "ewald_method") || return nothing

    settings = [("method", "Ewald", header["ewald_method"])]
    if lowercase(header["ewald_method"]) ∈ ("wolf", "zahn")
        append!(settings, [("rcut", "Ewald", header["ewald_rcut"]), ("alpha", "Ewald", header["ewald_alpha"])])
    end
    for (key, value) in header
        startswith(key, "ewald_qeff_") || continue
        type, element = (split(key[12:end], "_", limit=2)..., "")[1:2]   # ewald_qeff_H_N: qeff_N of H
        push!(settings, (isempty(element) ? "qeff" : "qeff_$element", type, value))
    end

    same(a, b) = (x = tryparse(Float64, string(a)); y = tryparse(Float64, string(b));
                  isnothing(x) || isnothing(y) ? lowercase(string(a)) == lowercase(string(b)) : isapprox(x, y, rtol=1e-10))
    adopted, differences = String[], String[]
    for (key, block, value) in settings
        name = block == "Ewald" ? key : "$(key) ($block)"
        if conf(key, block) == "default"
            set_value!(conf, key, block, value)
            push!(adopted, "$name = $value")
        elseif !same(conf(key, block), value)
            push!(differences, "$name: $(conf(key, block)) in the config, $value in $file")
        end
    end
    if !isempty(differences)
        error("The Ewald settings in the config differ from the parameter file $file, with which the model was fitted:\n  " *
              join(differences, "\n  ") * "\nRemove them from the config to use the settings of the file, or set " *
              "check_params = false in the Ewald block to use the config.")
    end
    if rank == 0 && verbosity > 0 && !isempty(adopted)
        println("Ewald settings from $file: ", join(adopted, ", "))
    end
    return nothing
end

"""
    init_ewald_params(coupling_labels, pol_labels, conf; species=UInt8[], initas=get_ewald_init_params(conf))

Initial parameters of the Ewald model: the coupling γ of each entry `(species, l)` of `coupling_labels` (`l = -1`: all
orbitals of the species, `coupling = species`; species 0: all species, `coupling = global`), followed by
the polarizability of each species in `pol_labels`. From the parameter file `initas` (e.g. the `params.dat` of a fit) if
given: `ewald_<element>_<l>` (`<l>` = s, p or d) for one orbital type, `ewald_<element>` for all orbital types of the
species, `polarizability_<element>`. Otherwise (and for entries missing in the file) from the config: `charge_scale_<l>`
of the atom block, else its `charge_scale`, else the `charge_scale` of the `Ewald` block; the tag `polarizability` of the
atom block. With `coupling = global`, the file must have the same coupling for all `species`.
"""
function init_ewald_params(coupling_labels, pol_labels, conf; species=UInt8[], initas=get_ewald_init_params(conf))
    γ = Float64[type == 0 ? get_ewald_charge_scale(conf) : get_charge_scale(conf, number_to_element(type), l) for (type, l) in coupling_labels]
    p = Float64[get_polarizability(conf, number_to_element(type)) for type in pol_labels]
    if initas ≠ "default"
        _, _, _, _, _, ewald_types, ewald_parameters, pol_types, pol_parameters = read_params(initas)
        file_γ = Dict(zip(ewald_types, ewald_parameters)); file_p = Dict(zip(pol_types, pol_parameters))
        γ = Float64[coupling_from_file(file_γ, label, γ[c], species, initas) for (c, label) in enumerate(coupling_labels)]
        p = Float64[get(file_p, number_to_element(type), p[t]) for (t, type) in enumerate(pol_labels)]
    end
    return vcat(γ, p)
end

"""
    coupling_from_file(file_γ, (type, l), default, species, file)

Coupling of the label `(type, l)` from the `ewald_` lines `file_γ` (type => value) of the parameter file `file`, `default`
if it has none. `l ≥ 0`: `ewald_<element>_<l>`, else `ewald_<element>`; `l < 0`: `ewald_<element>`, an error if the file only has
couplings per orbital type; species 0 (`coupling = global`): the value of all lines of the `species`, an error if they differ.
"""
function coupling_from_file(file_γ, (type, l), default, species, file)
    if type == 0
        names = number_to_element.(species)
        values = unique([v for (t, v) in file_γ if any(e -> t == e || startswith(t, e * "_"), names)])
        length(values) > 1 && error("$file has different Ewald couplings per species or orbital type: use coupling = species or orbital in the Ewald block.")
        return isempty(values) ? default : values[1]
    end
    element = number_to_element(type)
    l ≥ 0 && return get(file_γ, "$(element)_$("spdf"[l+1])", get(file_γ, element, default))
    if !haskey(file_γ, element) && any(x -> haskey(file_γ, "$(element)_$x"), "spd")
        error("$file has Ewald couplings per orbital type for $element (ewald_$(element)_<l>): use coupling = orbital in the Ewald block.")
    end
    return get(file_γ, element, default)
end

get_params(ewald::EwaldOnsites) = ewald.params

"""
    couplings(ewald::EwaldOnsites)

The couplings γ (one per entry of `ewald.coupling_labels`), a view of the parameters.
"""
couplings(ewald::EwaldOnsites) = view(ewald.params, 1:length(ewald.coupling_labels))

"""
    polarizabilities(ewald::EwaldOnsites)

The polarizabilities (Å³, one per entry of `ewald.pol_labels`), a view of the parameters.
"""
polarizabilities(ewald::EwaldOnsites) = view(ewald.params, length(ewald.coupling_labels)+1:length(ewald.params))

"""
    site_potentials(ewald::EwaldOnsites, index)

Electrostatic potential at the atoms of structure `index`: the potential of the effective charges plus that of the
induced dipoles, `φ + Φ p` (centered per species).
"""
site_potentials(ewald::EwaldOnsites, index) = ewald.potentials[index] .+ ewald.dipole_potentials[index] * polarizabilities(ewald)

"""
    update_mask(ewald::EwaldOnsites)

1 for the parameters that are updated in a fit (couplings with `update`, polarizabilities with `update_polarizability`), 0
for the others, whose gradient (also of the regularization) is zero.
"""
update_mask(ewald::EwaldOnsites) = vcat(fill(Float64(ewald.update), length(ewald.coupling_labels)),
                                        fill(Float64(ewald.update_polarizability), length(ewald.pol_labels)))

"""
    update!(ewald::EwaldOnsites, opt, dparams)

Update the Ewald onsite parameters in-place using the given optimizer
and parameter gradients `dparams`.
"""
function update!(ewald::EwaldOnsites, opt, dparams)
    update!(opt, ewald.params, dparams)
end

"""
    set_params!(ewald::EwaldOnsites, params)

Set the parameter vector of `ewald` in-place.

Throws an error if `params` does not match the expected size.
"""
function set_params!(ewald::EwaldOnsites, params)
    throw_error = size(ewald.params) ≠ size(params)
    if throw_error
        error("Parameter vector is not of correct size!")
    else
        ewald.params = params
    end
end

"""
    copy_params!(receiving_model::EwaldOnsites, sending_model::EwaldOnsites) -> Nothing

Copy matching parameters (couplings with the same species and orbital type, polarizabilities of the same species) from
one Ewald model to another.

# Arguments
- `receiving_model::EwaldOnsites`: The model whose parameters will be updated.
- `sending_model::EwaldOnsites`: The model providing parameter values.
"""
function copy_params!(receiving_model::EwaldOnsites, sending_model::EwaldOnsites)
    nr, ns = length(receiving_model.coupling_labels), length(sending_model.coupling_labels)
    for (i, label) in enumerate(sending_model.coupling_labels), (j, receiving_label) in enumerate(receiving_model.coupling_labels)
        label == receiving_label && (receiving_model.params[j] = sending_model.params[i])
    end
    for (i, label) in enumerate(sending_model.pol_labels), (j, receiving_label) in enumerate(receiving_model.pol_labels)
        label == receiving_label && (receiving_model.params[nr+j] = sending_model.params[ns+i])
    end
end

"""
    get_hr(ewald::EwaldOnsites, sp_mode, index; apply_soc=false)

Construct the onsite Hamiltonian contribution from Ewald potentials for
the structure at `index`.

Returns a vector of sparse matrices `Hr`, where only the on-site block
(`R = 0`) contains the diagonal electrostatic shifts `-γ_{s,l} (φ + Φ p)` and all other blocks
are zero.

# Arguments
- `sp_mode`: Flag for sparse mode (not relevant)
- `index`: Structure index
- `apply_soc`: If `true`, doubles the number of orbitals (spin degree of freedom)
"""
function get_hr(ewald::EwaldOnsites, sp_mode, index; apply_soc=false)
    φ = site_potentials(ewald, index)
    γ = couplings(ewald)
    c = ewald.coupling_per_strc[index]
    vals = Float64[]
    o = 0
    for (i, ϕ) in enumerate(φ), _ in 1:ewald.norb_per_strc[index][i]
        o += 1
        v = -γ[c[o]] * ϕ
        apply_soc ? append!(vals, (v, v)) : push!(vals, v)
    end

    is = collect(1:length(vals))
    Mewald = sparse(is, is, vals)
    Mzero = spzeros(ComplexF64, size(Mewald, 1), size(Mewald, 2))
    Hr = [ifelse(R == ewald.Rs_info[2, index], Mewald, Mzero) for R in 1:ewald.Rs_info[1, index]]
    return Hr
end

"""
    get_model_gradient(ewald::EwaldOnsites, indices, reg, dL_dHr; soc=true)

Gradient of the loss with respect to the couplings and polarizabilities. The onsite shift of orbital `o` (orbital type
`l`) of atom `i` (species `s`) is `Δε_o = -γ_{s,l} (φ_i + Σ_t p_t Φ_{i,t})`, so with `g_o = Re(dL/dHr[R=0]_oo)` (summed over
spin): `dL/dγ_{s,l} = -Σ g_o (φ_i + Σ_t p_t Φ_{i,t})` and `dL/dp_t = -Σ_o g_o γ_{s,l} Φ_{i,t}`. Zero (also without the
regularization term) for the parameters that are not updated (`update_mask`).
"""
function get_model_gradient(ewald::EwaldOnsites, indices, reg, dL_dHr; soc=true)
    mask = update_mask(ewald)
    dparams = zeros(length(ewald.params))
    any(!iszero, mask) || return dparams
    nc = length(ewald.coupling_labels); γ = couplings(ewald)
    for (n, index) in enumerate(indices)
        G = dL_dHr[n][ewald.Rs_info[2, index]]
        φ = site_potentials(ewald, index); Φ = ewald.dipole_potentials[index]; c = ewald.coupling_per_strc[index]
        o = 0
        for i in eachindex(φ), _ in 1:ewald.norb_per_strc[index][i]
            o += 1
            g = soc ? real(G[2o-1, 2o-1]) + real(G[2o, 2o]) : real(G[o, o])
            dparams[c[o]] -= g * φ[i]
            for t in axes(Φ, 2)
                dparams[nc+t] -= g * γ[c[o]] * Φ[i, t]
            end
        end
    end
    return mask .* (dparams + backward(reg, ewald.params))
end

"""
    write_params(ewald::EwaldOnsites, conf=get_empty_config(); filename="params.dat")

Add the Ewald couplings and the polarizabilities to the parameter file `filename` (written by the TB and SOC models
before), where they are read with `init_params = params.dat` in the `Ewald` block, and the Ewald settings
(`ewald_settings`) to its header. The coupling of a species is written as one `ewald_<element>` line if it is the same
for all its orbital types, else as `ewald_<element>_<l>` lines (`<l>` = s, p or d); polarizabilities as
`polarizability_<element>` lines.
"""
function write_params(ewald::EwaldOnsites, conf=get_empty_config(); filename="params.dat")
    isfile(filename) || return nothing
    parameters, parameter_values, ion_types, soc_parameters, _, _, _, _, _, exponent_labels, exponents = read_params(filename)
    γ = couplings(ewald); ewald_types = String[]; ewald_parameters = Float64[]
    if ewald.coupling_labels == [(0x00, -1)]
        append!(ewald_types, number_to_element.(ewald.species)); append!(ewald_parameters, fill(γ[1], length(ewald.species)))
    end
    for type in filter(≠(0), unique(first.(ewald.coupling_labels)))
        cs = findall(label -> label[1] == type, ewald.coupling_labels)
        element = number_to_element(type)
        if all(γ[c] == γ[cs[1]] for c in cs)
            push!(ewald_types, element); push!(ewald_parameters, γ[cs[1]])
        else
            append!(ewald_types, ["$(element)_$("spdf"[ewald.coupling_labels[c][2]+1])" for c in cs]); append!(ewald_parameters, γ[cs])
        end
    end
    write_params(parameters, parameter_values, ion_types, soc_parameters, conf, filename=replace(filename, r"\.dat$" => ""),
                 ewald_types=ewald_types, ewald_parameters=ewald_parameters, ewald_settings=ewald_settings(conf),
                 pol_types=number_to_element.(ewald.pol_labels), pol_parameters=collect(polarizabilities(ewald)),
                 exponent_labels=exponent_labels, exponents=exponents)
end
