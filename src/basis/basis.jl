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
    get_geometry_tensor(strc, basis, conf=get_empty_config(); tmethod=get_tmethod(conf), rcut=get_rcut(conf))

Constructs the geometry tensor based on the structure of the system, the orbital basis, and configuration settings.

# Arguments
- `strc`: The structure object, containing information about ions, lattice, and geometry of the system.
- `basis`: The basis object, containing the orbital information and overlap parameters.
- `conf`: Configuration object, defaults to an empty configuration if not provided.
- `tmethod`: Transformation method used for coordinate system alignment (default is obtained from the configuration).
- `rcut`: Cutoff distance for nearest neighbor interactions (default is obtained from the configuration).

# Keyword Arguments
- `tmethod`: The transformation method for aligning coordinate systems.
- `rcut`: The distance cutoff for nearest neighbor interactions.

# Returns
- The reshaped geometry tensor, represented as a matrix of sparse matrices, which encodes the overlap contributions for the system's orbital interactions for each parameter.
"""
function get_geometry_tensor(strc, basis, conf=get_empty_config();
                                file=nothing,
                                comm=nothing,
                                rank=0,
                                nranks=1,
                                npar=Threads.nthreads(),
                                tmethod=get_tmethod(conf), 
                                rcut=get_rcut(conf), 
                                sp_tol=get_sp_tol(conf), 
                                rcut_tol=get_rcut_tol(conf))

    norbs = length.(basis.orbitals)
    offset = cumsum(vcat(0, norbs[1:end-1]))
    ion_types = get_ion_types(strc.ions)
    nn_dict = get_nn_thresholds(strc.ions, frac_to_cart(strc.Rs, strc.lattice), strc.point_grid, conf)
    oc_dicts, mode_dicts = get_oc_and_mode_dicts(basis.overlaps, strc.ions)

    Ts = frac_to_cart(strc.Rs, strc.lattice)

    rllm_dict = get_rllm_from_file(basis.overlaps, conf, comm = comm, file=file)
    # Distance dependence and label of every overlap, looked up once instead of for every matrix element
    rllms = [rllm_dict[string(Cllm, apply_oc=true)] for Cllm in basis.overlaps]
    overlap_labels = [me_to_overlap_label(Cllm.type) for Cllm in basis.overlaps]
    # Whether an overlap contributes to a matrix element depends only on the ion types and the angular momenta
    # of the two orbitals, so the contributing overlaps are determined once per combination
    representative_orbitals = Dict{Tuple{Int64, Int64}, eltype(eltype(basis.orbitals))}()
    for iion in eachindex(basis.orbitals), orb in basis.orbitals[iion]
        get!(representative_orbitals, (Int64(ion_types[iion]), orb.type.l), orb)
    end
    # For neighbors (nnlabel ≠ 0), the parameter index depends only on the overlap and the nearest-neighbor label
    param_indices = Dict{Tuple{Int64, Int64}, Int64}()
    for k in eachindex(basis.overlaps), nnlabel in unique(Int64(param.nnlabel) for param in basis.parameters)
        nnlabel == 0 && continue
        param_indices[(k, nnlabel)] = get_param_index(basis.overlaps[k], nnlabel, basis.parameters, nothing, nothing, 0, 0, overlap_label=overlap_labels[k])
    end
    contributing_overlaps = Dict{NTuple{4, Int64}, Vector{Int64}}()
    for ((type1, l1), orb1) in representative_orbitals, ((type2, l2), orb2) in representative_orbitals
        ion_label = IonLabel(type1, type2, sorted=false)
        contributing_overlaps[(type1, type2, l1, l2)] = [k for k in eachindex(basis.overlaps)
            if overlap_contributes_to_matrix_element(basis.overlaps[k], orb1, orb2, ion_label)]
    end
    orbital_axes = [SVector{3, Float64}[orb.axis for orb in orbs] for orbs in basis.orbitals]
    orbital_ls = [Int64[orb.type.l for orb in orbs] for orbs in basis.orbitals]
    overlap_types = SVector{2, UInt8}[sort(Cllm.ion_label.types) for Cllm in basis.overlaps]
    onsite, sepNN = get_onsite(conf), get_sepNN(conf)
    parts = tmap(grid_cell_chunks(strc.point_grid, n=4 * npar)) do chunk
        entries = (v=Int64[], i=Int64[], j=Int64[], R=Int64[], val=Float64[])
        radial = zeros(length(basis.overlaps))
        angles = zeros(5)
        for (grid_point, inds) in chunk, nn_grid_point in nn_grid_points(grid_point, strc.point_grid.dictR),
            (iion2, R) in strc.point_grid.dictR[nn_grid_point], iion1 in inds
            ion_label = IonLabel(ion_types[iion1], ion_types[iion2], sorted=false)
            T⃗ = SVector{3, Float64}(Ts[1, R], Ts[2, R], Ts[3, R])
            r⃗₁ = strc.ions[iion1].pos - strc.ions[iion1].dist
            r⃗₂ = apply_pbc(strc.ions[iion2].pos, T⃗) - strc.ions[iion2].dist
            
            r_nd = normdiff(strc.ions[iion1].pos, strc.ions[iion2].pos, T⃗)
            r = normdiff(r⃗₁, r⃗₂)
            if r_nd ≤ rcut && r-abs(rcut_tol) < rcut && length(basis.orbitals[iion1]) > 0 && length(basis.orbitals[iion2]) > 0
                # Bond frames for both directions of the bond (the reversed one for swapped orbital pairs)
                Û = get_sk_transform_matrix(r⃗₁, r⃗₂, orbital_axes[iion1][1], tmethod)
                Û_swap = get_sk_transform_matrix(r⃗₂, r⃗₁, orbital_axes[iion1][1], tmethod)
                nnlabel = get_nn_label(r, nn_dict[ion_label], conf, onsite=onsite, sepNN=sepNN)
                # The radial part depends only on the overlap and the distance, not on the orbitals
                f = fcut(r, rcut, rcut_tol)
                types_sorted = sort(ion_label.types)
                for k in eachindex(basis.overlaps)
                    radial[k] = overlap_types[k] == types_sorted ? rllms[k](r) * f : 0.0
                end
                for jorb1 in eachindex(orbital_axes[iion1]), jorb2 in eachindex(orbital_axes[iion2])
                    i = offset[iion1] + jorb1
                    j = offset[iion2] + jorb2
                    orb1 = basis.orbitals[iion1][jorb1]; orb2 = basis.orbitals[iion2][jorb2]
                    # Hybrid orbitals (l < 0) are evaluated in the bond frame from atom 1 to atom 2, as before the d orbitals
                    hybrid = orb1.type.l < 0 || orb2.type.l < 0
                    bondswap = !hybrid && determine_bondswap(ion_types[iion1], ion_types[iion2], orb1.type, orb2.type, i, j)
                    U = bondswap ? Û_swap : Û
                    base = get_base_orb(orb1.type, orb2.type, bondswap=bondswap)
                    θ₁, φ₁ = get_rotated_angles(U, orbital_axes[iion1][jorb1])
                    θ₂, φ₂ = get_rotated_angles(U, orbital_axes[iion2][jorb2])

                    for k in contributing_overlaps[(Int64(ion_types[iion1]), Int64(ion_types[iion2]), orbital_ls[iion1][jorb1], orbital_ls[iion2][jorb2])]
                        Cllm = basis.overlaps[k]
                        mode = get_mode(mode_dicts, k, ion_label, jorb1, jorb2)
                        orbconfig = get_orbconfig(oc_dicts, k, ion_label, jorb1, jorb2)
                        # Base orbitals in the order of the parameter definition. Two signs of the bond direction (parity
                        # (-1)^(l₁ + l₂) for odd l₁ + l₂), which the angular functions of dorbs did not account for:
                        # - reversing the order of the base orbitals without reversing the bond;
                        # - parameters with decreasing l (e.g. Ga+As_psσ), whose distance dependence already contains
                        #   the sign (see distance_dependence), which the angular part applies again.
                        reversed_base = !hybrid && Cllm.type.base_ls ≠ (base[1].l, base[2].l)
                        base_k = reversed_base ? reverse(base) : base
                        odd = !hybrid && isodd(base[1].l + base[2].l)
                        flips = (reversed_base ? 1 : 0) + (Cllm.type.base_ls[1] > Cllm.type.base_ls[2] ? 1 : 0)
                        parity = odd && isodd(flips) ? -1.0 : 1.0
                        v::Int64 = nnlabel ≠ 0 ? get(param_indices, (k, nnlabel), 0) :
                            get_param_index(Cllm, nnlabel, basis.parameters, orb1, orb2, i, j, overlap_label=overlap_labels[k])

                        angles[1] = θ₁; angles[2] = φ₁; angles[3] = θ₂; angles[4] = φ₂
                        evaluate_overlap!(angles, Cllm, base_k, orbconfig, mode)
                        hval = parity * angles[5] * radial[k]

                        if abs(hval) ≥ sp_tol
                            push!(entries.v, v); push!(entries.i, i); push!(entries.j, j); push!(entries.R, R); push!(entries.val, hval)
                        end
                    end
                end
            end
        end
        entries
    end
    return assemble_geometry_tensor(parts, length(basis.parameters), size(strc.Rs, 2), length(basis)[1])
end

"""
    evaluate_overlap!(buffer, overlap, base, orbconfig, mode)

Writes the angular part of `overlap` with the base orbitals `base` for the angles `buffer[1:4]` (θ₁, φ₁, θ₂, φ₂) to
`buffer[5]`.
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
    reshape_geometry_tensor(h_dict, NV, NR, Nε)

Reshapes the geometry tensor from a dictionary `h_dict` into a matrix of sparse matrices.

# Arguments
- `h_dict::Dict`: A dictionary where keys are tuples of the form `(v, i, j, R)` representing
  the tensor indices, and values are the corresponding tensor entries.
- `NV::Int`: Number of rows (vectors) in the reshaped matrix.
- `NR::Int`: Number of columns (points) in the reshaped matrix.
- `Nε::Int`: Not used explicitly in this function but typically represents the number of strain components.

# Returns
- A `NV x NR` matrix `h_out` where each element is a sparse matrix (`SparseMatrixCSC{Float64, Int64}`) formed by the corresponding
  indices and values from the input dictionary `h_dict`.
"""
function reshape_geometry_tensor(h_dict, NV, NR, Nε)
    # Count the entries first, such that the index vectors can be allocated with their final size
    counts = zeros(Int64, NV, NR)
    for (v, _, _, R) in keys(h_dict)
        counts[v, R] += 1
    end
    is = [Vector{Int64}(undef, counts[v, R]) for v in 1:NV, R in 1:NR]
    js = [Vector{Int64}(undef, counts[v, R]) for v in 1:NV, R in 1:NR]
    vals = [Vector{Float64}(undef, counts[v, R]) for v in 1:NV, R in 1:NR]
    fill!(counts, 0)
    for (key, val) in h_dict
        v, i, j, R = key
        n = counts[v, R] += 1
        is[v, R][n] = i
        js[v, R][n] = j
        vals[v, R][n] = val
    end
    h_out = Matrix{SparseMatrixCSC{Float64, Int64}}(undef, NV, NR)
    for R in axes(h_out, 2), v in axes(h_out, 1)
        h_out[v, R] = sparse(is[v, R], js[v, R], vals[v, R], Nε, Nε)
    end
    return h_out
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