"""
    get_tb_descriptor(model, strc, conf)

Calculate the TB descriptor for a given a TB `model`, a structure `strc` and a TBConfig file `conf`.
"""
function get_tb_descriptor(h, V, strc::Structure, basis, conf::Config; rcut=get_ml_rcut(conf), rcut_tol=get_rcut_tol(conf), apply_distortion=get_apply_distortion(conf), 
    env_scale=get_env_scale(conf), apply_distance_distortion=get_apply_distance_distortion(conf), strc_scale=get_strc_scale(conf),
    Z_scale=get_Z_scale(conf), R_scale=get_R_scale(conf), overlap_scale=get_overlap_scale(conf),
    apply_orthogonality = get_ml_apply_orthogonality(conf))
    sim_params = get_sim_params(conf)
    #env_scale = env_scale/sim_params
    Nε = length(basis); Norb_per_ion = size(basis); NR = size(strc.Rs, 2)

    h_env = SparseMatrixCSC{SVector{9, Float64}, Int64}[spzeros(SVector{9, Float64}, Nε, Nε) for _ in 1:NR]

    env = get_environmental_descriptor(h, V, strc, basis, conf) .* env_scale

    rs_ion = get_ion_positions(strc.ions)
    Ts = frac_to_cart(strc.Rs, strc.lattice)

    ij_map = get_ion_orb_to_index_map(Norb_per_ion)
    l_map = [basis.orbitals[iion][iorb].type.l for iion in 1:length(Norb_per_ion) for iorb in 1:Norb_per_ion[iion]]

    is = [Int64[] for R in 1:NR]
    js = [Int64[] for R in 1:NR]
    vals = [SVector{9, Float64}[] for R in 1:NR]
    points = iterate_nn_grid_points(strc.point_grid)
    grouped = group_by_R(points, NR)
    tforeach( eachindex(grouped)) do R
        @views grouped_R = grouped[R]
        for (iion, jion) in grouped_R
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
                #Zs = [proton_to_position(strc.ions[iion].type), proton_to_position(strc.ions[jion].type)]
                Zs = [strc.ions[iion].type, strc.ions[jion].type]
                iaxis = basis.orbitals[iion][iorb].axis
                jaxis = basis.orbitals[jion][jorb].axis
                φ, θs = get_angular_descriptors(ri, rj, iaxis, jaxis)

                orbswap = decide_orbswap(itype, jtype, l_i, env[i], l_j, env[j])
                angleswap = (θs[1] > θs[2] && itype == jtype) || (orbswap && itype ≠ jtype)

                Zs = orbswap ? reverse(Zs) : Zs
                θs = angleswap ? reverse(θs) : θs

                orb_i_type = basis.orbitals[iion][iorb].type
                orb_j_type = basis.orbitals[jion][jorb].type
                
                same_ion = iion == jion 

                Overlap_ID = orbital_pairs_to_id[canonical_pair(lm_to_orbital_map[(orb_i_type.l, orb_i_type.m)], lm_to_orbital_map[(orb_j_type.l, orb_j_type.m)])]
                if same_ion
                    Overlap_ID += length(orbital_pairs) + 1
                end


                Overlap_ID *= overlap_scale
                Zs = Zs .* Z_scale
                Δr_in *= R_scale
                #Overlap_ID *= 100
                #Zs = Zs .* 100.0

                

                if apply_distortion || apply_distance_distortion
                    φ = φ / 2π * strc_scale
                    θs = @. θs / 2π * strc_scale
                end
                isorthogonal = decide_orthogonal(Δr, i, j, l_i, l_j; apply_orthogonality=apply_orthogonality)

                if Δr ≤ rcut && fcut(Δr_dist, rcut+rcut_tol) > 0 && !isorthogonal
                    ii, jj = orbswap ? (j, i) : (i, j)
                    push!(is[R], i); push!(js[R], j); push!(vals[R], SVector{9, Float64}([Overlap_ID, Zs[1], Zs[2], Δr_in, φ, θs[1], θs[2], env[ii], env[jj]]))
                end
            end
        end
    end
    tforeach( eachindex(collect(1:NR))) do R
    #@views for R in 1:NR
        h_env[R] = sparse(is[R], js[R], copy(vals[R]), Nε, Nε)
    end
    return h_env
end


"""
    decide_orthogonal(Δr, i, j, l_i, l_j; apply_orthogonality=false) -> Bool

Determine whether two atomic orbitals should be treated as orthogonal.

# Arguments
- `Δr::Real` : Distance between the centers of the two orbitals.  
- `i::Int` : Index of the first orbital.  
- `j::Int` : Index of the second orbital.  
- `l_i::Int` : Angular momentum quantum number (or orbital type) of the first orbital.  
- `l_j::Int` : Angular momentum quantum number (or orbital type) of the second orbital.  
- `apply_orthogonality::Bool` (keyword, default=false) : Whether to enforce atomic orthogonality rules.

# Returns
- `Bool` : `true` if the orbitals are considered orthogonal, `false` otherwise.
"""
function decide_orthogonal(Δr, i, j, l_i, l_j; apply_orthogonality=false)
    !apply_orthogonality && return false
    Δr != 0 && return false
    (l_i < 0 || l_j < 0) && return false
    return i != j
end

"""
    reshape_structure_descriptors(descriptors) -> Matrix{Float64}

Reshapes a nested structure of sparse descriptors into a dense matrix (to be used as input for kmeans).

# Arguments
- `descriptors`: A nested collection of sparse matrices representing structure descriptors.

# Returns
- A matrix (`Matrix{Float64}`) where each column corresponds to a flattened descriptor.
"""
function reshape_structure_descriptors(descriptors, counts, weight_factor)
    out = hcat([begin 
                    counts[n] += 1
                    Vector(descriptor) 
                end 
                for n in eachindex(descriptors) 
                for R in eachindex(descriptors[n]) 
                for (i, j, descriptor) 
                in zip(findnz(descriptors[n][R])...)]...)
    w = get_descriptor_weights(counts, weight_factor)
    return Float32.(vcat(out, w'))
end

function reshape_structure_descriptor_single_system(descriptors)
    out = hcat([Vector(descriptor)  for R in eachindex(descriptors) for (i, j, descriptor) in zip(findnz(descriptors[R])...)]...)
    w = ones(size(out, 2))
    return Float32.(vcat(out, w'))
end

function build_submatrices(
    X::AbstractMatrix,
    conf;
    Z_scale=get_Z_scale(conf),
    overlap_scale=get_overlap_scale(conf)
)
    d, n = size(X)

    # Pass 1: count rows per (k1,k2)
    nthreads_total = Threads.maxthreadid() 
    local_counts = [Dict{Tuple{Int,Int,Int},Int}() for _ in 1:nthreads_total]

    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        key = (
            round(Int, X[1,i] / overlap_scale),
            round(Int, X[2,i] / Z_scale),
            round(Int, X[3,i] / Z_scale)
        )
        local_counts[tid][key] = get(local_counts[tid], key, 0) + 1
    end

    # Merge counts
    counts = Dict{Tuple{Int,Int,Int},Int}()

    for lc in local_counts
        for (key, c) in lc
            counts[key] = get(counts, key, 0) + c
        end
    end

    # Preallocate submatrices
    submatrices = Dict{Tuple{Int,Int,Int},Matrix{eltype(X)}}()

    for (key, c) in counts
        submatrices[key] = Matrix{eltype(X)}(undef, d, c)
    end

    # Pass 2: fill matrices
    cursors = Dict(key => Threads.Atomic{Int}(1) for key in keys(counts))

    Threads.@threads for i in 1:n
        key = (
            round(Int, X[1,i] / overlap_scale),
            round(Int, X[2,i] / Z_scale),
            round(Int, X[3,i] / Z_scale)
        )

        col = Threads.atomic_add!(cursors[key], 1)
        submatrices[key][:, col] = X[:, i]
    end

    return submatrices
end

function sort_by_key(
    X::Vector{SVector{9,Float64}},
    weights::AbstractVector,
    conf;
    Z_scale=get_Z_scale(conf),
    overlap_scale=get_overlap_scale(conf)
)
    n = length(X)
    @assert length(weights) == n "weights must have one entry per element of X"

    nthreads_total = Threads.maxthreadid() 

    # Pass 1: count rows per (k1,k2,k3)
    local_counts = [Dict{Tuple{Int,Int,Int},Int}() for _ in 1:nthreads_total]
    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        p = X[i]
        key = (
            round(Int, p[1] / overlap_scale),
            round(Int, p[2] / Z_scale),
            round(Int, p[3] / Z_scale)
        )
        local_counts[tid][key] = get(local_counts[tid], key, 0) + 1
    end

    # Merge counts
    counts = Dict{Tuple{Int,Int,Int},Int}()
    for lc in local_counts
        for (key, c) in lc
            counts[key] = get(counts, key, 0) + c
        end
    end

    # Canonical ordering of keys and their offsets into Xsorted
    keys_sorted = sort(collect(keys(counts)))
    offsets = Dict{Tuple{Int,Int,Int},Int}()
    key_ranges = Dict{Tuple{Int,Int,Int},UnitRange{Int}}()
    let pos = 1
        for key in keys_sorted
            c = counts[key]
            offsets[key] = pos
            key_ranges[key] = pos:(pos + c - 1)
            pos += c
        end
    end

    Xsorted = Vector{SVector{9,Float64}}(undef, n)
    weights_sorted = Vector{eltype(weights)}(undef, n)

    # Pass 2: fill Xsorted and weights_sorted together, using atomic offsets per key
    offset_atomics = Dict(key => Threads.Atomic{Int}(offsets[key]) for key in keys(counts))
    Threads.@threads for i in 1:n
        p = X[i]
        key = (
            round(Int, p[1] / overlap_scale),
            round(Int, p[2] / Z_scale),
            round(Int, p[3] / Z_scale)
        )
        dest = Threads.atomic_add!(offset_atomics[key], 1)
        Xsorted[dest] = p
        weights_sorted[dest] = weights[i]
    end

    return Xsorted, weights_sorted, key_ranges
end

function check_consistency(data_points, params; key_ranges=nothing, verbose=true)
    ok = true

    # 1. Basic length match
    if length(data_points) != length(params)
        ok = false
        verbose && @warn "data_points/params length mismatch" len_dp=length(data_points) len_params=length(params)
    end

    if key_ranges === nothing
        verbose && ok && println("Consistency OK: $(length(data_points)) points, $(length(params)) params, lengths match.")
        return ok
    end

    # 2. key_ranges total coverage vs actual lengths
    total_range_len = sum(length(r) for r in values(key_ranges); init=0)
    if total_range_len != length(data_points)
        ok = false
        verbose && @warn "key_ranges total length doesn't match data_points" total_range_len=total_range_len len_dp=length(data_points)
    end
    if total_range_len != length(params)
        ok = false
        verbose && @warn "key_ranges total length doesn't match params" total_range_len=total_range_len len_params=length(params)
    end

    # 3. Ranges partition [1:n] cleanly (no gaps/overlaps)
    all_ranges = sort(collect(values(key_ranges)), by = first)
    expected_start = 1
    for r in all_ranges
        if first(r) != expected_start
            ok = false
            verbose && @warn "gap or overlap in key_ranges" range=r expected_start=expected_start
            expected_start = last(r) + 1
            continue
        end
        expected_start = last(r) + 1
    end
    n = length(data_points)
    if expected_start - 1 != n
        ok = false
        verbose && @warn "key_ranges don't cover full data_points length" covered=(expected_start - 1) total=n
    end

    # 4. Per-key: data_points[range] and params[range] have matching length (always true by construction, but check max index validity)
    max_idx = maximum(last(r) for r in values(key_ranges); init=0)
    if max_idx > n
        ok = false
        verbose && @warn "key_ranges index out of bounds for data_points/params" max_idx=max_idx n=n
    end

    verbose && ok && println("Consistency OK: $(length(key_ranges)) keys, $n points, ranges partition cleanly.")
    return ok
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
decide_orbswap(itype, jtype, l_i, env_i, l_j, env_j) = (itype == jtype && l_i > l_j) || (itype == jtype && l_i == l_j && env_i > env_j) || (itype > jtype)

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
function get_descriptor_weights(Np_per_strc, weight_factor = -1.0)
    f = weight_factor

    # --- Weight construction ---
    counts  = [Int(x) for x in Np_per_strc]
    offsets = cumsum(counts)
    N_total = offsets[end]
    w_strc  = Vector{Float64}(undef, N_total)

    tforeach(eachindex(w_strc)) do j
        k = searchsortedfirst(offsets, j)
        w_strc[j] = Float64(Np_per_strc[k])^f
    end
    w_strc ./= mean(w_strc)
    return w_strc
end

function calc_npoint_ncluster(descr_dict, Npoints, Ncluster, conf; alpha = get_alpha(conf), Nc_min = get_nc_min(conf), Nc_max = get_nc_max(conf))
    #alpha = 0.5
    N_key = length(keys(descr_dict))
    keys_list = []
    for key in keys(descr_dict)
        push!(keys_list, key)
        #println(keys_list[end])
    end
    N_key = length(keys_list)
    descr_weights = zeros(Float64, N_key)
    tforeach(eachindex(keys_list)) do n
        weights = fweights(descr_dict[keys_list[n]][end, :])
        descr = descr_dict[keys_list[n]][1:end-1,:]
        descr_weights[n] = sum([var(descr[i, :], weights) for i in 1:size(descr, 1)]) * alpha + sum(weights) * (1 - alpha)
    end
    descr_weights ./= sum(descr_weights)
    np_nc_dict = Dict{Tuple{Int, Int, Int}, Tuple{Int, Int}}()
    Np_total = 0
    Ncluster_total = 0
    for n in eachindex(keys_list)
        N_descr = size(descr_dict[keys_list[n]], 2)
        Np = ceil(Int,max(1, Npoints * descr_weights[n]))
        Nc = ceil(Int,max(1, Ncluster * descr_weights[n]))
        key_tuple = keys_list[n]
        key_overlap, key_Z1, key_Z2 = key_tuple[1], key_tuple[2], key_tuple[3]
        N_overlap_ids = length(orbital_pairs)
        same_ion = key_overlap > N_overlap_ids
        true_key_overlap = same_ion ? key_overlap - N_overlap_ids - 1 : key_overlap
        element_label_1 = elements[key_Z1].symbol
        element_label_2 = elements[key_Z2].symbol
        overlap_label = orbital_id_to_pairs[true_key_overlap]
        #Nc_min = 10
        #Nc_max = 200

        Nc = ceil(Int, max(Nc_min, N_descr * 0.1))
        Nc = ceil(Int, min(Nc_max, Nc))
        #Nc = 50
        Np = ceil(Int, Nc * 5)
        #Np = 250

        same_ion_label = same_ion ? "DI" : "NN"
        label = "$element_label_1-$element_label_2-$(overlap_label[1])-$(overlap_label[2])-$same_ion_label"
        @info "Overlap: $label, N_descr: $N_descr, Nc : $Nc"
        @assert Np >= Nc
        Np_total += Np
        Ncluster_total += Nc
        np_nc_dict[keys_list[n]] = (Np, Nc)
    end
    @info "Total Np requested: $Np_total, Total Ncluster requested: $Ncluster_total, Total number of keys: $N_key"
    return np_nc_dict
end

function sample_structure_descriptors(descriptors_w; Ncluster=1, Npoints=1, alpha=0.5, ml_sampling="random")
    #d × n
    unique_descriptors = descriptors_w[1:end-1, :]
    Np = size(unique_descriptors, 2)
    if Npoints < Np
        w_strc = descriptors_w[end, :]
        result = Logging.with_logger(NullLogger()) do
            kmeans(unique_descriptors, Ncluster; weights=w_strc)
        end
        indices = result.assignments
        centroids = result.centers

        cluster_sizes = zeros(Float64, Ncluster)
        for i in eachindex(indices)
            cluster_sizes[indices[i]] += w_strc[i]
        end
        cluster_sizes = ceil.(cluster_sizes)

        cluster_variances = [mean([normdiff(unique_descriptors[:, i], centroids[:, c]) for i in findall(x -> x == c, indices)]) for c in 1:Ncluster]

        nonzero_clusters = findall(s -> s != 0, cluster_sizes)
        cluster_ids = nonzero_clusters
        cluster_sizes = cluster_sizes[cluster_ids]
        cluster_variances = cluster_variances[cluster_ids]

        size_weights = cluster_sizes ./ sum(cluster_sizes)
        spread_weights = cluster_variances ./ sum(cluster_variances)
        final_weights = alpha .* size_weights + (1 - alpha) .* spread_weights
        final_weights ./= sum(final_weights)

        points_per_cluster = round.(Int, final_weights .* Npoints)
        points_per_cluster .= max.(1, points_per_cluster)
        points_per_cluster .= min.(cluster_sizes, points_per_cluster)

        diff = Npoints - sum(points_per_cluster)
        if diff != 0
            sorted_clusters = sortperm(final_weights, rev=true)
            for i in 1:abs(diff)
                idx = sorted_clusters[mod1(i, length(sorted_clusters))]
                points_per_cluster[idx] += sign(diff)
            end
        end

        selected_indices = Int64[]
        for (i, cid) in enumerate(cluster_ids)
            cluster_indices = findall(x -> x == cid, indices)
            num_to_take = min(points_per_cluster[i], length(cluster_indices))

            selected = Int64[]
            if ml_sampling[1] == 'r'
                selected = sample(cluster_indices, num_to_take, replace=false)
            elseif ml_sampling[1] == 'f'
                selected = farthest_point_sampling(unique_descriptors, cluster_indices, num_to_take)
            end

            append!(selected_indices, selected)
        end

        selected_indices = unique(selected_indices)

    end
    selected_indices = Npoints >= Np ? collect(1:Np) : selected_indices

    Random.seed!()

    return SVector{size(unique_descriptors, 1), Float64}[SVector{size(unique_descriptors, 1)}(unique_descriptors[:, index]) for index in selected_indices]
end
function sample_structure_descriptors_random(descriptors; Npoints=1)
    Random.seed!(1234)

    total_points = size(descriptors, 2)
    Npoints = min(Npoints, total_points)

    selected_indices = sample(1:total_points, Npoints; replace=false)

    Random.seed!()

    return SVector{size(descriptors, 1), Float64}[
        SVector{size(descriptors, 1)}(descriptors[:, i]) 
        for i in selected_indices
    ]
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

            # --- Parallel distance update ---
            tforeach(eachindex(cluster_indices)) do i
                d = normdiff(descriptors[:, cluster_indices[i]], descriptors[:, last])
                dists[i] = min(dists[i], d)
            end

            # --- Serial selection of farthest unselected point ---
            sorted_inds = sortperm(dists, rev=true)
            next = findfirst(i -> cluster_indices[i] ∉ selected, sorted_inds)
            push!(selected, cluster_indices[sorted_inds[next]])
        end
    end

    return selected
end