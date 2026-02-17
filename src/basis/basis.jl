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
                                comm=nothing,
                                rank=0,
                                nranks=1,
                                npar=Threads.nthreads(),
                                tmethod=get_tmethod(conf), 
                                rcut=get_rcut(conf), 
                                sp_tol=get_sp_tol(conf), 
                                rcut_tol=get_rcut_tol(conf))

    ij_map = get_ion_orb_to_index_map(length.(basis.orbitals))
    ion_types = get_ion_types(strc.ions)
    nn_dict = get_nn_thresholds(strc.ions, frac_to_cart(strc.Rs, strc.lattice), strc.point_grid, conf)
    hs = [Dict{Tuple{Int64, Int64, Int64, Int64}, Float64}() for _ in 1:npar]
    oc_dicts, mode_dicts = get_oc_and_mode_dicts(basis.overlaps, strc.ions)

    Ts = frac_to_cart(strc.Rs, strc.lattice)
    nn_grid_points = iterate_nn_grid_points(strc.point_grid)

    rllm_dict = get_rllm(basis.overlaps, conf, comm=comm, rank=rank, nranks=nranks)
    Threads.@threads for (chunk_id, indices) in enumerate(chunks(nn_grid_points, n=npar))
        for (iion1, iion2, R) in indices
            ion_label = IonLabel(ion_types[iion1], ion_types[iion2], sorted=false)
            r⃗₁ = strc.ions[iion1].pos - strc.ions[iion1].dist
            r⃗₂ = strc.ions[iion2].pos - strc.ions[iion2].dist - Ts[:, R]
            
            r_nd = normdiff(strc.ions[iion1].pos, strc.ions[iion2].pos .- Ts[:, R])
            r = normdiff(r⃗₁, r⃗₂)
            if r_nd ≤ rcut && r-abs(rcut_tol) < rcut && length(basis.orbitals[iion1]) > 0 && length(basis.orbitals[iion2]) > 0
                Û = get_sk_transform_matrix(r⃗₁, r⃗₂, basis.orbitals[iion1][1].axis, tmethod)
                nnlabel = get_nn_label(r, nn_dict[ion_label], conf)
                for jorb1 in eachindex(basis.orbitals[iion1]), jorb2 in eachindex(basis.orbitals[iion2])
                    orb1 = basis.orbitals[iion1][jorb1]
                    orb2 = basis.orbitals[iion2][jorb2]
                    i = ij_map[(iion1, jorb1)]
                    j = ij_map[(iion2, jorb2)]
                    θ₁, φ₁ = get_rotated_angles(Û, orb1.axis)
                    θ₂, φ₂ = get_rotated_angles(Û, orb2.axis)

                    for k in eachindex(basis.overlaps)
                        Cllm = basis.overlaps[k]
                        Rllm = rllm_dict[string(Cllm, apply_oc=true)]
                        mode = get_mode(mode_dicts, k, ion_label, jorb1, jorb2)
                        orbconfig = get_orbconfig(oc_dicts, k, ion_label, jorb1, jorb2)
                        if overlap_contributes_to_matrix_element(Cllm, orb1, orb2, ion_label)
                            v = get_param_index(Cllm, nnlabel, basis.parameters, orb1, orb2, i, j)

                            hval = Cllm(orbconfig, mode, θ₁, φ₁, θ₂, φ₂) * Rllm(r) * fcut(r, rcut, rcut_tol)

                            if haskey(hs[chunk_id], (v, i, j, R)) && abs(hval) ≥ sp_tol
                                hs[chunk_id][(v, i, j, R)] += hval
                            elseif abs(hval) ≥ sp_tol
                                hs[chunk_id][(v, i, j, R)] = hval
                            end
                        end
                    end
                end
            end
        end
    end
    h = merge(hs...)
    return reshape_geometry_tensor(h, length(basis.parameters), size(strc.Rs, 2), length(basis)[1])
end

"""
    overlap_contributes_to_matrix_element(overlap, orb1, orb2, ion_label)

Return true if `overlap` contributes to the matrix element between the orbitals `orb1` and `orb2` for `ion_label`.

# Arguments
- `overlap`: The overlap object, which contains information about the types of orbitals involved and the ion label.
- `orb1`: The first orbital.
- `orb2`: The second orbital.
- `ion_label`: The ion label to check if the overlap corresponds to the same type of ions as the orbitals.

# Returns
- `Bool`: Returns `true` if the overlap contributes to the matrix element, i.e., if the orbital types of `orb1` and `orb2` match the types in the overlap, and the `ion_label` matches.
"""
function overlap_contributes_to_matrix_element(overlap, orb1, orb2, ion_label)::Bool
    l₁, l₂ = orb1.type.l, orb2.type.l
    l₁_ov, l₂_ov = overlap.type.base[1].l, overlap.type.base[2].l
    return same_ion_label(overlap, ion_label) && ( (l₁ == l₁_ov && l₂ == l₂_ov) || (l₁ == l₂_ov && l₂ == l₁_ov))
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

# Returns
- `Int64`: The index `v` of the parameter in the `parameters` vector that matches the conditions. If no match is found, the function returns `0`.
"""
function get_param_index(overlap, nnlabel, parameters, orb1, orb2, i, j)::Int64
    if nnlabel ≠ 0
        for (v, param) in enumerate(parameters)
            if param.nnlabel == nnlabel && me_to_overlap_label(overlap.type) == param.overlap_label && same_ion_label(overlap, param.ion_label)
                return v
            end
        end
    else
        m = i == j ? 0 : 1
        for (v, param) in enumerate(parameters)
            same_overlap = [orb1.type.l, orb2.type.l, m] == param.overlap_label || [orb2.type.l, orb1.type.l, m] == param.overlap_label
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
    is = [Vector{Int64}() for _ in 1:NV, _ in 1:NR]
    js = [Vector{Int64}() for _ in 1:NV, _ in 1:NR]
    vals = [Vector{Float64}() for _ in 1:NV, _ in 1:NR]
    for (key, val) in h_dict
        v, i, j, R = key
        push!(is[v, R], i)
        push!(js[v, R], j)
        push!(vals[v, R], val)
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
              rcut_tol=get_rcut_tol(conf),
              npar=get_nthreads_bands(conf))

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
- `npar`: Number of parallel chunks used for bond generation.

# Returns
- `bonds::Vector{SparseMatrixCSC{SVector{3,Float64},Int64}}`:
  A vector of sparse matrices, one for each lattice translation `R`, whose
  nonzero entries store Cartesian bond vectors as `SVector{3,Float64}`.
"""
function get_bonds(strc, basis, conf=get_empty_config(); 
                            rcut=get_rcut(conf), 
                            rcut_tol=get_rcut_tol(conf),
                            npar=get_nthreads_bands(conf))

    nn_grid_points = iterate_nn_grid_points(strc.point_grid)
    Nε = length(basis)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    nR = size(strc.Rs, 2)
    ij_map = get_ion_orb_to_index_map(length.(basis.orbitals))

    # Allocate thread-local storage
    is_thread = [ [Int[] for _ in 1:nR] for _ in 1:npar ]
    js_thread = [ [Int[] for _ in 1:nR] for _ in 1:npar ]
    vals_x_thread = [ [Float64[] for _ in 1:nR] for _ in 1:npar ]
    vals_y_thread = [ [Float64[] for _ in 1:nR] for _ in 1:npar ]
    vals_z_thread = [ [Float64[] for _ in 1:nR] for _ in 1:npar ]

    Threads.@threads for (chunk_id, indices) in enumerate(chunks(nn_grid_points, n=npar))
        @views for (iion1, iion2, R) in indices
            r⃗₁ = strc.ions[iion1].pos - strc.ions[iion1].dist
            r⃗₂ = strc.ions[iion2].pos - strc.ions[iion2].dist - Ts[:, R]

            r_nd = normdiff(strc.ions[iion1].pos, strc.ions[iion2].pos .- Ts[:, R])
            r = normdiff(r⃗₁, r⃗₂)
            bond = SVector{3, Float64}(r⃗₂ .- r⃗₁)

            if r_nd ≤ rcut && r - abs(rcut_tol) < rcut &&
            !isempty(basis.orbitals[iion1]) && !isempty(basis.orbitals[iion2])

                for jorb1 in eachindex(basis.orbitals[iion1]), jorb2 in eachindex(basis.orbitals[iion2])
                    i = ij_map[(iion1, jorb1)]
                    j = ij_map[(iion2, jorb2)]

                    push!(is_thread[chunk_id][R], i)
                    push!(js_thread[chunk_id][R], j)
                    push!(vals_x_thread[chunk_id][R], bond[1])
                    push!(vals_y_thread[chunk_id][R], bond[2])
                    push!(vals_z_thread[chunk_id][R], bond[3])
                end
            end
        end
    end

    # Merge thread-local buffers
    is = [Int[] for _ in 1:nR]
    js = [Int[] for _ in 1:nR]
    vals_x = [Float64[] for _ in 1:nR]
    vals_y = [Float64[] for _ in 1:nR]
    vals_z = [Float64[] for _ in 1:nR]

    for R in 1:nR
        for t in 1:npar
            append!(is[R], is_thread[t][R])
            append!(js[R], js_thread[t][R])
            append!(vals_x[R], vals_x_thread[t][R])
            append!(vals_y[R], vals_y_thread[t][R])
            append!(vals_z[R], vals_z_thread[t][R])
        end
    end

    # Build sparse matrices
    bonds = Vector{NTuple{3,SparseMatrixCSC{Float64,Int64}}}(undef, nR)
    for R in 1:nR
        bx = sparse(is[R], js[R], vals_x[R], Nε, Nε)
        by = sparse(is[R], js[R], vals_y[R], Nε, Nε)
        bz = sparse(is[R], js[R], vals_z[R], Nε, Nε)
        bonds[R] = (bx, by, bz)
    end

    return bonds
end