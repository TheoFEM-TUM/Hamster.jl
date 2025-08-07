"""
    get_tb_descriptor(model, strc, conf)

Calculate the TB descriptor for a given a TB `model`, a structure `strc` and a TBConfig file `conf`.
"""
function get_tb_descriptor(h, V, strc::Structure, basis, conf::Config; rcut=get_ml_rcut(conf), apply_distortion=get_apply_distortion(conf), env_scale=get_env_scale(conf),
    apply_distance_distortion=get_apply_distance_distortion(conf), strc_scale=get_strc_scale(conf))

    Nε = length(basis); Norb_per_ion = size(basis); NR = size(strc.Rs, 2)

    h_env = SparseMatrixCSC{SVector{8, Float64}, Int64}[spzeros(SVector{8, Float64}, Nε, Nε) for _ in 1:NR]

    env = get_environmental_descriptor(h, V, strc, basis, conf)

    rs_ion = get_ion_positions(strc.ions)
    Ts = frac_to_cart(strc.Rs, strc.lattice)

    ij_map = get_ion_orb_to_index_map(Norb_per_ion)
    l_map = [basis.orbitals[iion][iorb].type.l for iion in 1:length(Norb_per_ion) for iorb in 1:Norb_per_ion[iion]]

    is = [Int64[] for R in 1:NR]
    js = [Int64[] for R in 1:NR]
    vals = [SVector{8, Float64}[] for R in 1:NR]
    for (iion, jion, R) in iterate_nn_grid_points(strc.point_grid)
        ri = rs_ion[iion]
        rj = rs_ion[jion] - Ts[:, R]
        Δr = normdiff(ri, rj)
        if apply_distortion
            ri -= strc.ions[iion].dist
            rj -= strc.ions[jion].dist
        end
        Δr_dist = normdiff(ri, rj)
        Δr_in = (apply_distance_distortion || apply_distortion) ? Δr_dist : Δr
        if apply_distortion || apply_distance_distortion
            Δr_in = Δr_in / rcut * strc_scale
        end
        for iorb in 1:Norb_per_ion[iion], jorb in 1:Norb_per_ion[jion]
            i = ij_map[(iion, iorb)]
            j = ij_map[(jion, jorb)]
            itype = strc.ions[iion].type; l_i = l_map[i]
            jtype = strc.ions[jion].type; l_j = l_map[j]
            
            Zs = [element_to_number(strc.ions[iion].type), element_to_number(strc.ions[jion].type)]
            iaxis = basis.orbitals[iion][iorb].axis
            jaxis = basis.orbitals[jion][jorb].axis
            φ, θs = get_angular_descriptors(ri, rj, iaxis, jaxis)

            orbswap = decide_orbswap(itype, jtype, l_i, env[i], l_j, env[j])
            angleswap = (θs[1] > θs[2] && itype == jtype) || (orbswap && itype ≠ jtype)

            Zs = orbswap ? reverse(Zs) : Zs
            θs = angleswap ? reverse(θs) : θs

            if apply_distortion || apply_distance_distortion
                φ = φ / 2π * strc_scale
                θs = @. θs / 2π * strc_scale
            end

            if Δr_dist ≤ rcut
                ii, jj = orbswap ? (j, i) : (i, j)
                push!(is[R], i); push!(js[R], j); push!(vals[R], SVector{8, Float64}([Zs[1], Zs[2], Δr_in, φ, θs[1], θs[2], env[ii] * env_scale, env[jj] * env_scale]))
            end
        end
    end
    @views for R in 1:NR
        h_env[R] = sparse(is[R], js[R], copy(vals[R]), Nε, Nε)
    end
    return h_env
end

"""
    reshape_structure_descriptors(descriptors) -> Matrix{Float64}

Reshapes a nested structure of sparse descriptors into a dense matrix (to be used as input for kmeans).

# Arguments
- `descriptors`: A nested collection of sparse matrices representing structure descriptors.

# Returns
- A matrix (`Matrix{Float64}`) where each column corresponds to a flattened descriptor.
"""
function reshape_structure_descriptors(descriptors)
    out = hcat([Vector(descriptor) for n in eachindex(descriptors) for R in eachindex(descriptors[n]) for (i, j, descriptor) in zip(findnz(descriptors[n][R])...)]...)
    return out
end

"""
    decide_orbswap(itype, jtype, l_i, m_i, l_j, m_j) -> Bool

Determines whether two orbitals should be swapped to enforce a consistent ordering, based on their associated ion types and quantum numbers.
The ordering is determined by:

1. Comparing element types using a periodic table-based numerical ordering (`element_to_number`).
2. If element types are the same, comparing orbital angular momentum quantum numbers (`l_i`, `l_j`).
3. If `l` values are equal, comparing magnetic quantum numbers (`m_i`, `m_j`).

This helps maintain consistent descriptor or feature vector construction in systems involving atomic orbitals.

# Arguments
- `itype`: Symbol or string representing the first ion type (e.g., `:H`, `"O"`).
- `jtype`: Symbol or string representing the second ion type.
- `l_i`: Orbital angular momentum quantum number of the first orbital.
- `m_i`: Magnetic quantum number of the first orbital.
- `l_j`: Orbital angular momentum quantum number of the second orbital.
- `m_j`: Magnetic quantum number of the second orbital.

# Returns
- `true` if the orbitals should be swapped to maintain ordering; `false` otherwise.
"""
decide_orbswap(itype, jtype, l_i, env_i, l_j, env_j) = (itype == jtype && l_i > l_j) || (itype == jtype && l_i == l_j && env_i > env_j) || (element_to_number(itype) > element_to_number(jtype))

"""
    get_angular_descriptors(itype, jtype, ri, rj, iaxis, jaxis, orbswap)

Computes angular descriptors based on the relative positions and orbital orientations of two atoms.

# Arguments
- `itype, jtype`: Atomic types of the two atoms.
- `ri, rj`: Position vectors of the two atoms.
- `iaxis, jaxis`: Axes defining the local orbital orientation for each atom.

# Returns
- `φ::Float64`: The angle between the two orbital axes.
- `θs::Vector{Float64}`: A sorted or conditionally reversed list of angles between each axis and the bond direction.

# Behavior
- Computes the normalized bond direction `Δrij` and `Δrji` depending on the distance between the atoms.
- Determines the angle `φ` between the two orbital axes.
- Computes `θs`, the angles between each axis (`iaxis`, `jaxis`) and the respective bond directions.
- Ensures consistent ordering of `θs` based on atomic types and orbital swapping rules.
"""
function get_angular_descriptors(ri, rj, iaxis, jaxis)
    Δr = normdiff(ri, rj)
    Δrij = Δr > 0 ? normalize(rj - ri) : normalize(iaxis)
    Δrji = Δr > 0 ? normalize(ri - rj) : normalize(jaxis)
    φ = calc_angle(iaxis, jaxis)
    θs = Float64[calc_angle(iaxis, Δrij), calc_angle(jaxis, Δrji)]
    return φ, θs
end

"""
    get_environmental_descriptor(h, V, strc, basis, conf::Config; apply_params=false, rcut=get_ml_rcut(conf))

Computes the environmental descriptor for a given structure, basis, and configuration object.

# Arguments
- `h`: The geometry tensor for the given structure.
- `V`: A vector of interaction parameters.
- `strc`: The structure containing lattice and atomic positions.
- `basis`: The basis set defining the system's orbitals.
- `conf::Config`: Configuration object.

# Keyword Arguments
- `apply_params`: If `true`, uses `V` as given; otherwise, is set to ones.
- `rcut`: Cut-off radius for interactions, defaults to `get_ml_rcut(conf)`.

# Returns
- `env`: A vector representing the environmental descriptor.
"""
function get_environmental_descriptor(h, V, strc, basis, conf::Config; apply_params=false, rcut=get_ml_rcut(conf))
    Vapp = apply_params ? V : ones(length(V))
    Nε = length(basis); Norb = size(basis)

    Hr = get_hr(h, Vapp, Sparse())
    env = zeros(Nε)

    R0 = findR0(strc.Rs)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    rs_ion = get_ion_positions(strc.ions)
    index_map = get_index_to_ion_orb_map(Norb)
    @views for (R, H) in enumerate(Hr)
        for (i, j, Hval) in zip(findnz(H)...)
            iion, _ = index_map[i]
            jion, _ = index_map[j]
            Δr = normdiff(rs_ion[iion], rs_ion[jion], Ts[:, R])
            if (iion ≠ jion && R == R0) || (R ≠ R0)
                @inbounds env[i] += Hval * fcut(Δr, rcut)
            end
        end
    end
    return env
end

"""
    sample_structure_descriptors(descriptors; Ncluster=1, Npoints=1, alpha=0.5, ml_sampling="random")

Selects a subset of descriptor vectors using K-Means clustering, weighted by cluster size and spread.

# Arguments
- `descriptors`: A matrix where each column represents a descriptor vector.
- `Ncluster::Int=1`: The number of clusters for K-Means.
- `Npoints::Int=1`: The total number of descriptor vectors to select.
- `alpha::Float64=0.5`: A weighting factor (0 ≤ α ≤ 1) that balances selection between cluster size (α → 1) and spread (α → 0).
- `ml_sampling::String`: Determines how points are selected from each cluster. Defaults to random.

# Returns
- A matrix of selected descriptor vectors with `Npoints` columns.
"""
function sample_structure_descriptors(descriptors; Ncluster=1, Npoints=1, alpha=0.5, ml_sampling="random")
    result = kmeans(descriptors, Ncluster)
    indices = result.assignments
    centroids = result.centers
        
    cluster_sizes = [count(x -> x == c, indices) for c in 1:Ncluster]
    cluster_variances = [mean([normdiff(descriptors[:, i], centroids[:, c]) for i in findall(x -> x == c, indices)]) for c in 1:Ncluster]

    # Filter empty clusters
    nonzero_clusters = findall(s -> s ≠ 0, cluster_sizes)
    cluster_sizes = cluster_sizes[nonzero_clusters]
    cluster_variances = cluster_variances[nonzero_clusters]

    # Compute weights
    size_weights = cluster_sizes ./ sum(cluster_sizes)
    spread_weights = cluster_variances ./ sum(cluster_variances)
    final_weights = alpha .* size_weights + (1 - alpha) .* spread_weights
    final_weights ./= sum(final_weights)  # Normalize

    points_per_cluster = round.(Int, final_weights .* Npoints)
    points_per_cluster .= max.(1, points_per_cluster)

    # Adjust to ensure the exact number of `Npoints` is selected
    diff = Npoints - sum(points_per_cluster)
    if diff != 0
        sorted_clusters = sortperm(final_weights, rev=true)
        for i in 1:abs(diff)
            points_per_cluster[sorted_clusters[i]] += sign(diff)
        end
    end

    selected_indices = Int64[]
    for c in eachindex(cluster_sizes)
        cluster_indices = findall(x -> x == c, indices)
        num_to_take = min(points_per_cluster[c], length(cluster_indices))
        
        selected = Int64[]
        if ml_sampling[1] == 'r'
            selected = sample(cluster_indices, num_to_take, replace=false)
        elseif ml_sampling[1] == 'f'
            selected = farthest_point_sampling(descriptors, cluster_indices, num_to_take)
        end
        append!(selected_indices, selected)
    end
    summary = (nz_clusters = length(cluster_sizes), cluster_sizes = cluster_sizes, points_per_cluster = points_per_cluster, cluster_variances = cluster_variances)

    return SVector{size(descriptors, 1), Float64}[SVector{size(descriptors, 1)}(descriptors[:, index]) for index in selected_indices]
end

"""
    farthest_point_sampling(descriptors, cluster_indices, num_to_take)

Selects `num_to_take` diverse points from a subset of data specified by `cluster_indices` using
greedy farthest-point sampling based on Euclidean distance.

# Arguments
- `descriptors::AbstractMatrix{<:Real}`: A matrix of feature vectors where each column corresponds to a data point.
- `cluster_indices::Vector{Int}`: Indices of the points in `descriptors` that belong to the cluster to sample from.
- `num_to_take::Int`: Number of points to select.

# Returns
- `selected::Vector{Int}`: Indices of the selected points (subset of `cluster_indices`) representing a diverse subset.
"""
function farthest_point_sampling(descriptors, cluster_indices, num_to_take)
    cluster_size = length(cluster_indices)
    selected = Int[]

    if cluster_size > 0 && num_to_take > 0
        dists = fill(Inf, cluster_size)

        push!(selected, rand(cluster_indices))
        while length(selected) < num_to_take
            last = selected[end]
            for (i, point_index) in enumerate(cluster_indices)
                d = normdiff(descriptors[:, point_index], descriptors[:, last])
                dists[i] = min(dists[i], d)
            end

            sorted_inds = sortperm(dists, rev=true)
            next = findfirst(i->i∉selected, cluster_indices[sorted_inds])
            push!(selected, cluster_indices[sorted_inds][next])
        end
    end

    return selected
end