
mutable struct Kernelpoints{K<:Tuple}
    datapoints :: Vector{SVector{8, Float64}}
    weights :: Vector{Int64}
    keys :: Vector{K}
    key_ranges :: Dict{K, UnitRange{Int}}
    key_sizes  :: Dict{K, Int64}
end

function get_sorted_Kernelpoints(data_points::Vector{SVector{8, Float64}}, weights = ones(Int64, length(data_points)), params = zeros(Float64, length(data_points)), conf = get_empty_config())
    Xsorted, weights_sorted, params_sorted, key_ranges = sort_by_key(data_points, weights, params, conf)
    ks = sort(collect(keys(key_ranges)))
    sizes = [length(key_ranges[k]) for k in ks]
    key_sizes = Dict(zip(ks, sizes))
    return Kernelpoints(Xsorted, weights_sorted, ks, key_ranges, key_sizes), params_sorted
end

function sort_by_key(
    X::Vector{SVector{8,Float64}},
    weights::Vector{Int64},
    params::Vector{Float64},
    conf = get_empty_config();
    key_dims = get_ml_key_dims(conf)
)
    n = length(X)
    @assert length(weights) == n "weights must have one entry per element of X"

    nthreads_total = Threads.maxthreadid()

    nkeys = max(length(key_dims), 1)
    KeyT = NTuple{nkeys,Int}

    keyfun = if isempty(key_dims)
        p -> (0,)
    else
        p -> ntuple(j -> round(Int, p[key_dims[j]]), nkeys)
    end

    # Pass 1: count rows per key, per thread (order-independent merge)
    local_counts = [Dict{KeyT,Int}() for _ in 1:nthreads_total]
    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        key = keyfun(X[i])
        local_counts[tid][key] = get(local_counts[tid], key, 0) + 1
    end

    counts = Dict{KeyT,Int}()
    for lc in local_counts
        for (key, c) in lc
            counts[key] = get(counts, key, 0) + c
        end
    end

    # Canonical key ordering -> fixed, nthreads-independent bucket ranges
    keys_sorted = sort(collect(keys(counts)))
    offsets = Dict{KeyT,Int}()
    key_ranges = Dict{KeyT,UnitRange{Int}}()
    let pos = 1
        for key in keys_sorted
            c = counts[key]
            offsets[key] = pos
            key_ranges[key] = pos:(pos + c - 1)
            pos += c
        end
    end

    # Pass 2: scatter into buckets using atomics. Intra-bucket order here
    # is racy/nthreads-dependent, but that's fixed up in Pass 3 below.
    Xsorted = Vector{SVector{8,Float64}}(undef, n)
    weights_sorted = Vector{eltype(weights)}(undef, n)
    params_sorted = Vector{eltype(params)}(undef, n)
    orig_idx = Vector{Int}(undef, n)  # track original index per slot

    offset_atomics = Dict(key => Threads.Atomic{Int}(offsets[key]) for key in keys_sorted)
    Threads.@threads for i in 1:n
        key = keyfun(X[i])
        dest = Threads.atomic_add!(offset_atomics[key], 1)
        Xsorted[dest] = X[i]
        weights_sorted[dest] = weights[i]
        params_sorted[dest] = params[i]
        orig_idx[dest] = i
    end

    # Pass 3: deterministic fixup - within each bucket, reorder by original
    # index. This is the step that makes the result independent of
    # nthreads and scheduling: bucket *contents* were already fixed by
    # `key_ranges`, and now bucket *order* is fixed too.
    Threads.@threads for key in keys_sorted
        r = key_ranges[key]
        length(r) <= 1 && continue
        perm = sortperm(view(orig_idx, r))
        if !issorted(view(orig_idx, r))
            Xsorted[r] = Xsorted[r][perm]
            weights_sorted[r] = weights_sorted[r][perm]
            params_sorted[r] = params_sorted[r][perm]
        end
    end

    return Xsorted, weights_sorted, params_sorted, key_ranges
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
function verify_kernelpoints(kp::Kernelpoints, conf = get_empty_config();
                              key_dims = get_ml_key_dims(conf))
    n = length(kp.datapoints)
    ok = true

    nkeys = max(length(key_dims), 1)
    KeyT = NTuple{nkeys,Int}

    keyfun = if isempty(key_dims)
        p -> (0,)
    else
        p -> ntuple(j -> round(Int, p[key_dims[j]]), nkeys)
    end

    # --- Check 2: every key in key_ranges is in keys, and vice versa ---
    keyset_from_keys = Set(kp.keys)
    keyset_from_ranges = Set(keys(kp.key_ranges))
    if keyset_from_keys != keyset_from_ranges
        only_in_keys = setdiff(keyset_from_keys, keyset_from_ranges)
        only_in_ranges = setdiff(keyset_from_ranges, keyset_from_keys)
        if !isempty(only_in_keys)
            @warn "keys present in `keys` but not in `key_ranges`" only_in_keys
        end
        if !isempty(only_in_ranges)
            @warn "keys present in `key_ranges` but not in `keys`" only_in_ranges
        end
        ok = false
    end

    # --- Check 3: ranges partition 1:n exactly (no gaps, no overlaps) ---
    all_ranges = sort(collect(values(kp.key_ranges)); by = first)
    expected_start = 1
    for r in all_ranges
        if first(r) != expected_start
            @warn "gap or overlap in key_ranges before this range" range=r expected_start
            ok = false
        end
        expected_start = last(r) + 1
    end
    if expected_start - 1 != n
        @warn "ranges do not cover all datapoints" covered=(expected_start - 1) n
        ok = false
    end

    # --- Check 4: every datapoint's recomputed key matches the key owning its index ---
    # Build index -> key lookup from key_ranges
    idx_to_key = Vector{Union{Nothing,KeyT}}(undef, n)
    fill!(idx_to_key, nothing)
    for (k, r) in kp.key_ranges
        for i in r
            if idx_to_key[i] !== nothing
                @warn "index assigned to multiple keys" index=i key1=idx_to_key[i] key2=k
                ok = false
            end
            idx_to_key[i] = k
        end
    end

    for i in 1:n
        p = kp.datapoints[i]
        true_key = keyfun(p)
        assigned_key = idx_to_key[i]
        if assigned_key === nothing
            @warn "datapoint index not covered by any key_range" index=i
            ok = false
        elseif assigned_key != true_key
            @warn "datapoint key mismatch" index=i assigned=assigned_key recomputed=true_key
            ok = false
        end
    end

    return ok
end

struct SimMat{T1, T2}
    feature_vec :: Vector{T1}
    feature_shape :: Tuple{Vector{T2}, Int64}
    sim_params :: Float64
end



"""
    get_kernel_features(structure_descriptors, data_points, sim_params, tol = 1e-8) -> Vector{T3}, Tuple{Vector{T2}, Int64}
Generates kernel feature vectors based on structure descriptors and data points.
# Arguments
- `structure_descriptors`: A collection of structure descriptors.
- `data_points`: A collection of data points.
- `sim_params`: Parameters for the similarity function.
- `tol`: Tolerance for filtering small values (default = 1e-8).
"""


function get_kernel_features(structure_descriptors, kp, sim_params, tol = 0.1; conf = get_empty_config(), rank = 0, systems = nothing, key_dims = get_ml_key_dims(conf))
    #dim_weights = get_dim_weights(conf)
    data_points = kp.datapoints
    key_ranges = kp.key_ranges

    nkeys = max(length(key_dims), 1)
    KeyT = NTuple{nkeys,Int64}

    keyfun = if isempty(key_dims)
        hin -> (0,)
    else
        hin -> ntuple(j -> Int64(hin[key_dims[j]]), nkeys)
    end

    N_mats = size(structure_descriptors)[1]
    systems = systems === nothing ? [string("system_", i) for i in 1:N_mats] : systems
    N_dp = size(data_points)[1]
    descr_sizes = [(size(structure_descriptors[i])[1], size(structure_descriptors[i][1])[1]) for i in 1:N_mats]
    Desc_Vec = [
        [ begin
            n = nnz(structure_descriptors[i][R])
            (Vector{Tuple{SparseVector{Float64, Int64}, Tuple{Int64, Int64, KeyT}}}(undef, n), n)
        end
        for R in 1:descr_sizes[i][1] ]
        for i in 1:N_mats
        ]
    N_test = zeros(Int64, N_mats)
    N_total = zeros(Int64, N_mats)
    for i in 1:N_mats
        N_R, Ne = descr_sizes[i]
        for R in 1:N_R
            h_env_R = structure_descriptors[i][R]
            touples = collect(zip(findnz(h_env_R)...))
            N_total_temp  = length(touples)
            N_test_temp = zeros(N_total_temp)
            tforeach(1:length(touples)) do n
                i_mat, j_mat, hin = touples[n]
                key = keyfun(hin)
                data_points_vector = @view data_points[key_ranges[key]]
                #val_vec = exp_sim_all(data_points_vector, hin, σ=sim_params, dim_weights = dim_weights)
                val_vec = exp_sim_all(data_points_vector, hin, σ=sim_params)
                val_vec[abs.(val_vec) .<= tol] .= 0
                val_vec = sparse(val_vec)
                covered = nnz(val_vec) > 0 ? 1 : 0
                N_test_temp[n] = covered
                Desc_Vec[i][R][1][n] = (val_vec,(i_mat,j_mat, key))
            end
            N_test[i] += sum(N_test_temp)
            N_total[i] += N_total_temp
        end
        if get_hyperopt_niter(conf) < 2; @info "Rank $rank: Finished kernel features for mat $(systems[i]) Nr. ($i / $N_mats) with Ncovered = ( $(N_test[i]) / $(N_total[i]) ) || $(ceil(Int, N_test[i] / N_total[i] * 100 )) %"; end
    end
    structure_descriptors = nothing
    GC.gc()
    return SimMat(Desc_Vec, (descr_sizes, N_dp), sim_params)
end


exp_sim2(x1::SVector{8,Float64}, x2::SVector{8,Float64}; σ=√0.05, dim_weights = ones(8)) =
    exp(-normdiff(x1.*dim_weights, x2.*dim_weights)^2 / (2σ^2))

exp_sim_all(x1, x2::SVector{8,Float64}; σ=√0.05, dim_weights = ones(8)) = exp_sim2.(x1, Ref(x2); σ=σ, dim_weights=dim_weights)