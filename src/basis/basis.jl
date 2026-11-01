"""
    struct Basis{Orb, Ov, P, R}

A data structure that represents the basis of a system, containing information about orbitals, overlap integrals, parameters, and precomputed radial functions (RLLM).

# Fields
- `orbitals::Orb`: A collection of orbitals for each ion in the system. This typically contains the orbital functions and axes associated with each ion.
- `overlaps::Ov`: A collection of overlap integrals between orbitals. These overlap integrals define how the orbitals interact with each other spatially.
- `parameters::P`: Parameters used for computing the matrix elements between orbitals. These can include interaction strengths, angular momentum values, and other system-specific constants.
- `rllm::R`: Precomputed radial functions (RLLM), typically stored as spline objects for efficient interpolation. These functions depend on the distance between ions and are used in overlap integrals.
"""
struct Basis{Orb, Ov, P}
    orbitals :: Orb
    overlaps :: Ov
    parameters :: P
end

"""
    Basis(strc::Structure, conf=get_empty_config()) -> Basis

Construct a `Basis` object from a given `Structure` and configuration.

# Arguments
- `strc::Structure`: The `Structure` object representing the atomic or molecular structure for which the basis is being constructed.
- `conf`: (Optional) The configuration object that holds parameters to control the basis construction. Defaults to `get_empty_config()`.

# Returns
- `Basis`: A `Basis` object.
"""
function Basis(strc::Structure, conf=get_empty_config(); comm=nothing)
    orbitals = get_orbitals(strc, conf)
    overlaps = get_overlaps(strc.ions, orbitals, conf)
    parameters = get_parameters_from_overlaps(overlaps, conf)
    return Basis(orbitals, overlaps, parameters)
end

"""
    length(basis::Basis) -> Int

Compute the total number of orbitals in the `basis`.

# Arguments
- `basis::Basis`: The `Basis` object containing orbitals for each ion in the system.

# Returns
- `Int`: The total number of orbitals, calculated by summing the lengths of the orbital sets for all ions in the `basis`.
"""
Base.length(basis::Basis) = sum(length.(basis.orbitals))

"""
    Base.size(basis::Basis) -> Tuple{Int}

Returns the size of the `Basis` object, which corresponds to the number of orbitals centered on each ion.

# Arguments
- `basis::Basis`: The basis object containing a set of orbitals.

# Returns
- A tuple containing the number of orbitals in the basis.
"""
Base.size(basis::Basis) = length.(basis.orbitals)

"""
    nparams(basis::Basis) -> Int

Return the number of TB overlap parameters defined in `basis`.

# Arguments
- `basis::Basis`: The `Basis` object containing orbitals for each ion in the system.

# Returns
- `Int`: The total number of parameters.
"""
nparams(basis::Basis) = length(basis.parameters)

"""
    get_geometry_tensor(strc, basis, conf=get_empty_config(); file=nothing, comm=nothing, npar=Threads.nthreads(),
                        tmethod=get_tmethod(conf), rcut=get_rcut(conf), sp_tol=get_sp_tol(conf),
                        rcut_tol=get_rcut_tol(conf))

Geometry tensor of the structure `strc` in the basis `basis`: a matrix `h[v, R]` of sparse `Nε × Nε` matrices with the
contribution of parameter `v` to the Hamiltonian between the home cell and the translation `R`, such that
`H(R) = Σᵥ pᵥ h[v, R]`.

A pair of ions is included if its distance is within `rcut` for the reference positions (`pos`) and within
`rcut + |rcut_tol|` for the distorted positions (`pos - dist`). The matrix elements use the distorted positions.

# Keyword Arguments
- `file`, `comm`: HDF5 file and MPI communicator to read the radial functions (rllm) from.
- `npar`: number of threads; the grid cells are split into `4 npar` chunks.
- `tmethod`: method for the Slater-Koster transformation matrix.
- `rcut`, `rcut_tol`: cutoff radius and width of the smooth cutoff (above `rcut` if positive, below if negative).
- `sp_tol`: entries with a smaller absolute value are dropped.
"""
function get_geometry_tensor(strc, basis, conf=get_empty_config();
                             file=nothing,
                             comm=nothing,
                             npar=Threads.nthreads(),
                             tmethod=get_tmethod(conf),
                             rcut=get_rcut(conf),
                             sp_tol=get_sp_tol(conf),
                             rcut_tol=get_rcut_tol(conf))
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    lookup = geometry_tensor_lookup(strc, basis, conf, file=file, comm=comm)
    nn_cutoffs = get_nn_cutoffs(strc.ions, Ts, strc.point_grid, conf)
    onsite, sepNN = get_onsite(conf), get_sepNN(conf)

    parts = tmap(grid_cell_chunks(strc.point_grid, n=4 * npar)) do chunk
        entries = (v=Int64[], i=Int64[], j=Int64[], R=Int64[], val=Float64[])
        radial = zeros(length(basis.overlaps))
        angles = zeros(5)
        for (grid_point, inds) in chunk, nn_grid_point in nn_grid_points(grid_point, strc.point_grid.dictR),
            (iion2, R) in strc.point_grid.dictR[nn_grid_point], iion1 in inds

            (isempty(basis.orbitals[iion1]) || isempty(basis.orbitals[iion2])) && continue
            ion1, ion2 = strc.ions[iion1], strc.ions[iion2]
            T⃗ = SVector{3, Float64}(Ts[1, R], Ts[2, R], Ts[3, R])
            r⃗₁ = ion1.pos - ion1.dist
            r⃗₂ = apply_pbc(ion2.pos, T⃗) - ion2.dist
            r = normdiff(r⃗₁, r⃗₂)
            (normdiff(ion1.pos, ion2.pos, T⃗) ≤ rcut && r - abs(rcut_tol) < rcut) || continue

            ion_label = IonLabel(lookup.ion_types[iion1], lookup.ion_types[iion2], sorted=false)
            nnlabel = get_nn_label(r, nn_cutoffs[ion_label], conf, onsite=onsite, sepNN=sepNN)
            radial_parts!(radial, lookup, ion_label, r, fcut(r, rcut, rcut_tol))
            add_bond_entries!(entries, angles, lookup, basis, iion1, iion2, R, r⃗₁, r⃗₂, nnlabel, radial, tmethod, sp_tol)
        end
        entries
    end
    return assemble_geometry_tensor(parts, length(basis.parameters), size(strc.Rs, 2), length(basis))
end

"""
    geometry_tensor_lookup(strc, basis, conf; file=nothing, comm=nothing)

Everything `get_geometry_tensor` needs per bond that depends only on the basis and the ion types, computed once per
structure:
- `ion_types`, `offset`: type of each ion and index of its first orbital minus one;
- `orbital_axes`, `orbital_ls`: axis and angular momentum of every orbital of every ion;
- `rllms`, `overlap_types`: radial function and sorted ion types of every overlap;
- `overlap_labels`, `param_indices`: see `get_neighbor_param_indices`;
- `contributing_overlaps`: see `get_contributing_overlaps`;
- `oc_dicts`, `mode_dicts`: see `get_oc_and_mode_dicts`.
"""
function geometry_tensor_lookup(strc, basis, conf; file=nothing, comm=nothing)
    ion_types = get_ion_types(strc.ions)
    rllm_dict = get_rllm_from_file(basis.overlaps, conf, comm=comm, file=file)
    overlap_labels = [me_to_overlap_label(overlap.type) for overlap in basis.overlaps]
    oc_dicts, mode_dicts = get_oc_and_mode_dicts(basis.overlaps, strc.ions)
    return (ion_types = ion_types,
            offset = cumsum(vcat(0, length.(basis.orbitals)[1:end-1])),
            orbital_axes = [SVector{3, Float64}[orb.axis for orb in orbs] for orbs in basis.orbitals],
            orbital_ls = [Int64[orb.type.l for orb in orbs] for orbs in basis.orbitals],
            rllms = [rllm_dict[string(overlap, apply_oc=true)] for overlap in basis.overlaps],
            overlap_types = SVector{2, UInt8}[sort(overlap.ion_label.types) for overlap in basis.overlaps],
            overlap_labels = overlap_labels,
            param_indices = get_neighbor_param_indices(basis, overlap_labels),
            contributing_overlaps = get_contributing_overlaps(basis, ion_types),
            oc_dicts = oc_dicts,
            mode_dicts = mode_dicts)
end

"""
    get_neighbor_param_indices(basis, overlap_labels)

Parameter index for every overlap `k` and nearest-neighbor label `nnlabel ≠ 0` as a dictionary `(k, nnlabel) => v`.
Unlike on-site parameters, these do not depend on the orbitals.
"""
function get_neighbor_param_indices(basis, overlap_labels)
    param_indices = Dict{Tuple{Int64, Int64}, Int64}()
    for k in eachindex(basis.overlaps), nnlabel in unique(Int64(param.nnlabel) for param in basis.parameters)
        nnlabel == 0 && continue
        param_indices[(k, nnlabel)] = get_param_index(basis.overlaps[k], nnlabel, basis.parameters, nothing, nothing,
                                                      0, 0, overlap_label=overlap_labels[k])
    end
    return param_indices
end

"""
    get_contributing_overlaps(basis, ion_types)

Indices of the overlaps that contribute to the matrix element between two orbitals as a dictionary
`(ion_type1, ion_type2, l1, l2) => ks`. This depends only on the ion types and angular momenta of the two orbitals
(see `overlap_contributes_to_matrix_element`).
"""
function get_contributing_overlaps(basis, ion_types)
    representative_orbitals = Dict{Tuple{Int64, Int64}, eltype(eltype(basis.orbitals))}()
    for iion in eachindex(basis.orbitals), orb in basis.orbitals[iion]
        get!(representative_orbitals, (Int64(ion_types[iion]), orb.type.l), orb)
    end
    contributing_overlaps = Dict{NTuple{4, Int64}, Vector{Int64}}()
    for ((type1, l1), orb1) in representative_orbitals, ((type2, l2), orb2) in representative_orbitals
        ion_label = IonLabel(type1, type2, sorted=false)
        contributing_overlaps[(type1, type2, l1, l2)] = [k for k in eachindex(basis.overlaps)
            if overlap_contributes_to_matrix_element(basis.overlaps[k], orb1, orb2, ion_label)]
    end
    return contributing_overlaps
end

"""
    radial_parts!(radial, lookup, ion_label, r, f)

Writes the radial function of every overlap at distance `r`, times the cutoff function value `f`, to `radial`
(zero for overlaps of other ion types).
"""
function radial_parts!(radial, lookup, ion_label, r, f)
    types = sort(ion_label.types)
    for k in eachindex(radial)
        radial[k] = lookup.overlap_types[k] == types ? lookup.rllms[k](r) * f : 0.0
    end
    return radial
end

"""
    add_bond_entries!(entries, angles, lookup, basis, iion1, iion2, R, r⃗₁, r⃗₂, nnlabel, radial, tmethod, sp_tol)

Appends the geometry tensor entries `(v, i, j, R, val)` of all orbital pairs of the bond from ion `iion1` at `r⃗₁` to
ion `iion2` at `r⃗₂` (translation `R`) to `entries`. `radial` are the radial parts from `radial_parts!` and `angles` is
a buffer of length 5 for `evaluate_overlap!`.
"""
function add_bond_entries!(entries, angles, lookup, basis, iion1, iion2, R, r⃗₁, r⃗₂, nnlabel, radial, tmethod, sp_tol)
    type1, type2 = lookup.ion_types[iion1], lookup.ion_types[iion2]
    ion_label = IonLabel(type1, type2, sorted=false)
    axes1, axes2 = lookup.orbital_axes[iion1], lookup.orbital_axes[iion2]
    Û = get_sk_transform_matrix(r⃗₁, r⃗₂, axes1[1], tmethod)
    Û_swap = get_sk_transform_matrix(r⃗₂, r⃗₁, axes1[1], tmethod)

    for jorb1 in eachindex(axes1), jorb2 in eachindex(axes2)
        i, j = lookup.offset[iion1] + jorb1, lookup.offset[iion2] + jorb2
        orb1, orb2 = basis.orbitals[iion1][jorb1], basis.orbitals[iion2][jorb2]
        # Hybrid orbitals (l < 0) are always evaluated in the bond frame from ion 1 to ion 2
        hybrid = orb1.type.l < 0 || orb2.type.l < 0
        bondswap = !hybrid && determine_bondswap(type1, type2, orb1.type, orb2.type, i, j)
        base = get_base_orb(orb1.type, orb2.type, bondswap=bondswap)
        U = bondswap ? Û_swap : Û
        angles[1], angles[2] = get_rotated_angles(U, axes1[jorb1])
        angles[3], angles[4] = get_rotated_angles(U, axes2[jorb2])

        ls = (Int64(type1), Int64(type2), lookup.orbital_ls[iion1][jorb1], lookup.orbital_ls[iion2][jorb2])
        for k in lookup.contributing_overlaps[ls]
            overlap = basis.overlaps[k]
            base_k, parity = base_in_parameter_order(overlap, base, hybrid)
            orbconfig = get_orbconfig(lookup.oc_dicts, k, ion_label, jorb1, jorb2)
            mode = get_mode(lookup.mode_dicts, k, ion_label, jorb1, jorb2)
            evaluate_overlap!(angles, overlap, base_k, orbconfig, mode)
            val = parity * angles[5] * radial[k]
            abs(val) ≥ sp_tol || continue

            v = nnlabel ≠ 0 ? get(lookup.param_indices, (k, nnlabel), 0) :
                get_param_index(overlap, nnlabel, basis.parameters, orb1, orb2, i, j, overlap_label=lookup.overlap_labels[k])
            push!(entries.v, v); push!(entries.i, i); push!(entries.j, j); push!(entries.R, R); push!(entries.val, val)
        end
    end
    return entries
end

"""
    base_in_parameter_order(overlap, base, hybrid)

The base orbitals `base` in the order of the parameter definition of `overlap`, and the sign of the angular part.

For odd `l₁ + l₂`, the sign changes with the bond direction (parity `(-1)^(l₁ + l₂)`), which the angular functions do
not account for. This happens when the base orbitals are reversed without reversing the bond, and for parameters with
decreasing `l` (e.g. `Ga+As_psσ`), whose distance dependence already contains the sign (see `distance_dependence`).
Hybrid orbitals keep their order and sign.
"""
function base_in_parameter_order(overlap, base, hybrid)
    hybrid && return base, 1.0
    reversed = overlap.type.base_ls ≠ (base[1].l, base[2].l)
    decreasing = overlap.type.base_ls[1] > overlap.type.base_ls[2]
    parity = isodd(base[1].l + base[2].l) && reversed ⊻ decreasing ? -1.0 : 1.0
    return reversed ? reverse(base) : base, parity
end

"""
    evaluate_overlap!(buffer, overlap, base, orbconfig, mode)

Writes the angular part of `overlap` with the base orbitals `base` for the angles `buffer[1:4]` (θ₁, φ₁, θ₂, φ₂) to
`buffer[5]`. The types of `overlap`, `orbconfig` and `mode` are only known at run time; writing the result to a buffer
instead of returning it avoids allocations.
"""
function evaluate_overlap!(buffer, overlap, base, orbconfig, mode)
    @inbounds buffer[5] = overlap(base, orbconfig, mode, buffer[1], buffer[2], buffer[3], buffer[4])
    return nothing
end

"""
    assemble_geometry_tensor(parts, NV, NR, Nε)

Sparse matrices `h[v, R]` from the entries collected by the tasks.
"""
function assemble_geometry_tensor(parts, NV, NR, Nε)
    counts = zeros(Int64, NV, NR)
    for part in parts, (v, R) in zip(part.v, part.R)
        counts[v, R] += 1
    end
    is = [Vector{Int64}(undef, counts[v, R]) for v in 1:NV, R in 1:NR]
    js = [Vector{Int64}(undef, counts[v, R]) for v in 1:NV, R in 1:NR]
    vals = [Vector{Float64}(undef, counts[v, R]) for v in 1:NV, R in 1:NR]
    fill!(counts, 0)
    for part in parts, n in eachindex(part.v)
        v, R = part.v[n], part.R[n]
        m = counts[v, R] += 1
        is[v, R][m] = part.i[n]; js[v, R][m] = part.j[n]; vals[v, R][m] = part.val[n]
    end
    h_out = Matrix{SparseMatrixCSC{Float64, Int64}}(undef, NV, NR)
    tforeach(CartesianIndices(h_out)) do I
        h_out[I] = sparse(is[I], js[I], vals[I], Nε, Nε)
    end
    return h_out
end

"""
    determine_bondswap(ion_type1, ion_type2, orb1, orb2, i, j)

Determine whether a bond-direction flip is required for a TB matrix element
given the definition of that interaction.

This accounts for the fact that some hopping parameters (e.g. s-p couplings)
change sign when the bonding axis is defined in the opposite direction.

Returns `true` if a bond swap is needed, which occurs when:
  - The two ions appear in reversed order (`areswapped(ion_type1, ion_type2)`), or
  - The ions are of the same type, but different orbital types (`orb1`, `orb2`)
    are used with reversed orbital indices (`i > j`).

Otherwise, returns `false`.
"""
function determine_bondswap(ion_type1, ion_type2, orb1, orb2, i, j)
    ionswap = areswapped(ion_type1, ion_type2)
    sameion_bondflip = ion_type1 == ion_type2 && i > j && typeof(orb1) ≠ typeof(orb2)
    return ionswap || sameion_bondflip
end

"""
    overlap_contributes_to_matrix_element(overlap, orb1, orb2, ion_label)

Return true if `overlap` contributes to the matrix element between the orbitals `orb1` and `orb2` for `ion_label`.

# Arguments
- `overlap`: The overlap object, which contains information about the types of orbitals involved and the ion label.
- `orb1`: The first orbital.
- `orb2`: The second orbital.
- `ion_label`: The ion label to check if the overlap corresponds to the same type of ions as the orbitals.

Between different elements, the angular momenta of the overlap belong to its elements in order (e.g. `Ga+As_spσ`: s on
Ga, p on As), so the reversed assignment (p on Ga, s on As) does not contribute. For the same element, both orders
contribute.

# Returns
- `Bool`: Returns `true` if the overlap contributes to the matrix element, i.e., if the orbital types of `orb1` and `orb2` match the types in the overlap, and the `ion_label` matches.
"""
function overlap_contributes_to_matrix_element(overlap, orb1, orb2, ion_label)::Bool
    l₁, l₂ = orb1.type.l, orb2.type.l
    l₁_ov, l₂_ov = overlap.type.base_ls[1], overlap.type.base_ls[2]
    same_ion_label(overlap, ion_label) || return false
    if ion_label.types[1] ≠ ion_label.types[2]
        ion_label.types[1] == overlap.ion_label.types[1] && return l₁ == l₁_ov && l₂ == l₂_ov
        return l₁ == l₂_ov && l₂ == l₁_ov
    end
    return (l₁ == l₁_ov && l₂ == l₂_ov) || (l₁ == l₂_ov && l₂ == l₁_ov)
end

"""
    get_param_index(overlap, nnlabel, parameters, orb1, orb2, i, j) :: Int64

Retrieve the index of a parameter that matches a given nearest neighbor label, overlap label, 
orbital types, and ion labels. 

# Arguments
- `overlap`: An object representing the overlap between two orbitals or ions.
- `nnlabel::Int`: The nearest neighbor (NN) label, identifying if this is a nearest neighbor interaction.
- `parameters::Vector`: A vector of parameter objects, each of which contains attributes such as `nnlabel`, `overlap_label`, and `ion_label`.
- `orb1`: The first orbital involved in the overlap.
- `orb2`: The second orbital involved in the overlap.
- `i::Int`: The index of the first orbital.
- `j::Int`: The index of the second orbital.
- `overlap_label`: (Optional) The overlap label of `overlap`, if already known (avoids recomputing it).

# Returns
- `Int64`: The index `v` of the parameter in the `parameters` vector that matches the conditions. If no match is found, the function returns `0`.
"""
function get_param_index(overlap, nnlabel, parameters, orb1, orb2, i, j; overlap_label=nnlabel ≠ 0 ? me_to_overlap_label(overlap.type) : nothing)::Int64
    if nnlabel ≠ 0
        for (v, param) in enumerate(parameters)
            if param.nnlabel == nnlabel && overlap_label == param.overlap_label && same_ion_label(overlap, param.ion_label)
                return v
            end
        end
    else
        m = i == j ? 0 : 1
        label₁ = SVector{3, Int64}(orb1.type.l, orb2.type.l, m)
        label₂ = SVector{3, Int64}(orb2.type.l, orb1.type.l, m)
        for (v, param) in enumerate(parameters)
            same_overlap = label₁ == param.overlap_label || label₂ == param.overlap_label
            if param.nnlabel == nnlabel && same_overlap && same_ion_label(overlap, param.ion_label)
                return v
            end
        end
        return v
    end
    return 0
end

"""
    get_oc_and_mode_dicts(overlaps, ions)

Generate dictionaries that map ion pairs to their respective orbital configuration (oc) 
and mode types for each overlap configuration.

# Arguments
- `overlaps::Vector`: A collection of overlap configurations, where each element represents a different overlap calculation.
- `ions::Vector`: A vector of ion types, where each ion has a `type` field that identifies the specific ion type.

# Returns
- `oc_dicts::Vector{Dict{IonLabel, Union{SymOrb, DefOrb, MirrOrb}}}`: A vector of dictionaries, each corresponding to one overlap configuration, mapping ion pairs (represented by `IonLabel`) to their respective orbital configuration (`SymOrb`, `DefOrb`, or `MirrOrb`).
- `mode_dicts::Vector{Dict{IonLabel, Union{NormalMode, ConjugateMode}}}`: A vector of dictionaries, each corresponding to one overlap configuration, mapping ion pairs (represented by `IonLabel`) to their respective mode type (`NormalMode` or `ConjugateMode`).
"""
function get_oc_and_mode_dicts(overlaps, ions)
    ion_types = unique([ion.type for ion in ions])
    unique_ion_labels = IonLabel[]
    for ion1 in ion_types, ion2 in ion_types
        push!(unique_ion_labels, IonLabel(ion1, ion2, sorted=false))
    end
    unique_ion_labels = unique(unique_ion_labels)
    oc_dicts = Dict{IonLabel, Union{SymOrb, DefOrb, MirrOrb}}[]
    mode_dicts = Dict{IonLabel, Union{NormalMode, ConjugateMode}}[]
    for kparam in eachindex(overlaps)
        oc_dict = Dict{IonLabel, Union{SymOrb, DefOrb, MirrOrb}}()
        mode_dict = Dict{IonLabel, Union{NormalMode, ConjugateMode}}()
        for ion_label in unique_ion_labels
            oc_dict[ion_label], mode_dict[ion_label] = decide_orbconfig(overlaps[kparam], ion_label)
        end
        push!(oc_dicts, oc_dict)
        push!(mode_dicts, mode_dict)
    end
    return oc_dicts, mode_dicts
end

"""
    get_bonds(strc, basis, conf=get_empty_config();
              rcut=get_rcut(conf),
              rcut_tol=get_rcut_tol(conf))

Construct bond vectors between basis orbitals for all lattice translations.

This function computes inter-ionic bond displacement vectors within a cutoff
radius and assembles them into sparse matrices indexed by basis orbitals.
For each lattice translation `R`, a sparse matrix of size `Nε × Nε` is returned,
where `Nε = length(basis)` and each nonzero entry stores the Cartesian bond
vector connecting a pair of orbitals.

# Arguments
- `strc`: Structure object containing ion positions, lattice vectors, and
  lattice translations.
- `basis`: Orbital basis associated with the ions in `strc`.
- `conf`: Configuration object.

# Keyword Arguments
- `rcut`: Real-space cutoff radius for bond construction.
- `rcut_tol`: Tolerance applied to the cutoff condition.

# Returns
- `bonds::Vector{NTuple{3, SparseMatrixCSC{Float64, Int64}}}`: for each lattice
  translation `R`, the x, y and z components of the Cartesian bond vectors
  r_j + R - r_i (sparse `Nε × Nε` matrices with the same sparsity pattern).
"""
function get_bonds(strc, basis, conf=get_empty_config();
                   rcut=get_rcut(conf),
                   rcut_tol=get_rcut_tol(conf))
    Nε = length(basis)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    nR = size(strc.Rs, 2)
    offset = cumsum(vcat(0, length.(basis.orbitals)[1:end-1]))

    parts = tmap(grid_cell_chunks(strc.point_grid)) do chunk
        is = [Int64[] for _ in 1:nR]; js = [Int64[] for _ in 1:nR]; ds = [SVector{3, Float64}[] for _ in 1:nR]
        for (grid_point, inds) in chunk, nn_grid_point in nn_grid_points(grid_point, strc.point_grid.dictR),
            (iion2, R) in strc.point_grid.dictR[nn_grid_point], iion1 in inds
            
            (isempty(basis.orbitals[iion1]) || isempty(basis.orbitals[iion2])) && continue
            
            r⃗₁ = strc.ions[iion1].pos - strc.ions[iion1].dist
            r⃗₂ = apply_pbc(strc.ions[iion2].pos, Ts[:, R]) - strc.ions[iion2].dist
            bond = SVector{3, Float64}(r⃗₂ .- r⃗₁)
            
            (normdiff(strc.ions[iion1].pos, strc.ions[iion2].pos, Ts[:, R]) ≤ rcut && norm(bond) - abs(rcut_tol) < rcut) || continue
            
            for jorb1 in eachindex(basis.orbitals[iion1]), jorb2 in eachindex(basis.orbitals[iion2])
                push!(is[R], offset[iion1] + jorb1); push!(js[R], offset[iion2] + jorb2); push!(ds[R], bond)
            end
        end
        (is, js, ds)
    end

    return tmap(1:nR) do R
        I = reduce(vcat, [p[1][R] for p in parts]); J = reduce(vcat, [p[2][R] for p in parts])
        D = reduce(vcat, [p[3][R] for p in parts])
        order = sparse(I, J, collect(1:length(I)), Nε, Nε)
        perm = nonzeros(order)
        ntuple(a -> SparseMatrixCSC(Nε, Nε, copy(order.colptr), copy(rowvals(order)), [D[n][a] for n in perm]), 3)
    end
end

get_unique_overlaps(bases) = unique(vcat([basis.overlaps for basis in bases]...))