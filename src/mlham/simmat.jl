
"""
    mutable struct Kernelpoints{K<:Tuple}

A structure containing sorted kernel points and associated metadata for efficient parameter organization.

# Fields
- `datapoints :: Vector{SVector{9, Float64}}`: Sorted vector of 9-dimensional data points representing kernel sampling points.
- `weights :: Vector{Int64}`: Weight associated with each data point.
- `keys :: Vector{K}`: List of unique keys (tuples) representing buckets of similar parameters.
- `key_ranges :: Dict{K, UnitRange{Int}}`: Mapping from keys to index ranges in the sorted datapoints array.
- `key_sizes :: Dict{K, Int64}`: Number of data points in each key bucket.
"""
mutable struct Kernelpoints{K<:Tuple}
    datapoints :: Vector{SVector{9, Float64}}
    weights :: Vector{Int64}
    keys :: Vector{K}
    key_ranges :: Dict{K, UnitRange{Int}}
    key_sizes  :: Dict{K, Int64}
end

"""
    get_sorted_Kernelpoints(data_points, weights, params, conf)

Creates a `Kernelpoints` structure by sorting data points into buckets based on their keys.

# Arguments
- `data_points::Vector{SVector{9, Float64}}`: Input data points to sort.
- `weights::Vector{Int64}`: Weights for each data point (default: `ones(Int64, length(data_points))`).
- `params::Vector{Float64}`: Parameters associated with data points (default: `zeros(Float64, length(data_points))`).
- `conf`: Configuration object containing key dimension specifications (default: `get_empty_config()`).

# Returns
- `kp::Kernelpoints`: Sorted kernel points structure with organized buckets.
- `params_sorted::Vector{Float64}`: Parameters reordered to match sorted data points.

# Notes
- Data points are partitioned into buckets using key dimensions from configuration.
- The sorting ensures deterministic ordering independent of thread scheduling.
"""
function get_sorted_Kernelpoints(data_points::Vector{SVector{9, Float64}}, weights = ones(Int64, length(data_points)), params = zeros(Float64, length(data_points)), conf = get_empty_config())
    Xsorted, weights_sorted, params_sorted, key_ranges = sort_by_key(data_points, weights, params, conf)
    ks = sort(collect(keys(key_ranges)))
    sizes = [length(key_ranges[k]) for k in ks]
    key_sizes = Dict(zip(ks, sizes))
    return Kernelpoints(Xsorted, weights_sorted, ks, key_ranges, key_sizes), params_sorted
end

"""
    sort_by_key(X, weights, params, conf; key_dims)

Sorts data points, weights, and parameters into buckets based on key dimensions using a thread-safe algorithm.

# Arguments
- `X::Vector{SVector{9,Float64}}`: Input data points to sort.
- `weights::Vector{Int64}`: Weights corresponding to each data point.
- `params::Vector{Float64}`: Parameters corresponding to each data point.
- `conf`: Configuration object (default: `get_empty_config()`).
- `key_dims`: Indices of dimensions to use for key computation (from configuration).
- `test_resort`: Boolean flag to force re-sorting within buckets for testing purposes (default: `false`).

# Returns
- `Xsorted::Vector{SVector{9,Float64}}`: Sorted data points partitioned by key.
- `weights_sorted::Vector{Int64}`: Weights reordered to match sorted data points.
- `params_sorted::Vector{Float64}`: Parameters reordered to match sorted data points.
- `key_ranges::Dict`: Mapping from keys to index ranges in sorted arrays.

# Notes
- Uses a three-pass algorithm: count rows per key, scatter into buckets using atomics, then deterministically reorder within buckets.
- This approach is thread-safe and produces deterministic results independent of thread scheduling.
"""
function sort_by_key(
    X::Vector{SVector{9,Float64}},
    weights::Vector{Int64},
    params::Vector{Float64},
    conf = get_empty_config();
    key_dims = get_key_dims(conf)
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
    Xsorted = Vector{SVector{9,Float64}}(undef, n)
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

        Xsorted[r] = Xsorted[r][perm]
        weights_sorted[r] = weights_sorted[r][perm]
        params_sorted[r] = params_sorted[r][perm]

    end

    return Xsorted, weights_sorted, params_sorted, key_ranges
end

"""
    check_consistency(data_points, params; key_ranges, verbose)

Verifies consistency between data points, parameters, and key ranges.

# Arguments
- `data_points`: Collection of data points.
- `params::Vector{Float64}`: Parameter vector to check.
- `key_ranges::Dict`: Optional mapping of keys to index ranges (default: `nothing`).
- `verbose::Bool`: Enable detailed console output (default: `true`).

# Returns
- `ok::Bool`: `true` if all consistency checks pass, `false` otherwise.

# Checks
1. Data points and parameters have matching length.
2. Key ranges total coverage matches actual lengths.
3. Key ranges partition indices with no gaps or overlaps.
4. All range indices are within bounds.
"""
function check_consistency(data_points, params; key_ranges=nothing, verbose=true)
    ok = true
    issues = String[]

    # 1. Basic length match
    if length(data_points) != length(params)
        ok = false
        push!(issues, "data_points/params length mismatch: len_dp=$(length(data_points)) len_params=$(length(params))")
    end

    if key_ranges === nothing
        if verbose
            if ok
                println("Consistency OK: $(length(data_points)) points, $(length(params)) params, lengths match.")
            else
                @warn "Consistency check failed:\n" * join(issues, "\n")
            end
        end
        return ok
    end

    # 2. key_ranges total coverage vs actual lengths
    total_range_len = sum(length(r) for r in values(key_ranges); init=0)
    if total_range_len != length(data_points)
        ok = false
        push!(issues, "key_ranges total length doesn't match data_points: total_range_len=$total_range_len len_dp=$(length(data_points))")
    end
    if total_range_len != length(params)
        ok = false
        push!(issues, "key_ranges total length doesn't match params: total_range_len=$total_range_len len_params=$(length(params))")
    end

    # 3. Ranges partition [1:n] cleanly (no gaps/overlaps)
    all_ranges = sort(collect(values(key_ranges)), by = first)
    expected_start = 1
    for r in all_ranges
        if first(r) != expected_start
            ok = false
            push!(issues, "gap or overlap in key_ranges: range=$r expected_start=$expected_start")
        end
        expected_start = last(r) + 1
    end
    n = length(data_points)
    if expected_start - 1 != n
        ok = false
        push!(issues, "key_ranges don't cover full data_points length: covered=$(expected_start - 1) total=$n")
    end

    # 4. Per-key: max index validity
    max_idx = maximum(last(r) for r in values(key_ranges); init=0)
    if max_idx > n
        ok = false
        push!(issues, "key_ranges index out of bounds for data_points/params: max_idx=$max_idx n=$n")
    end

    if verbose
        if ok
            println("Consistency OK: $(length(key_ranges)) keys, $n points, ranges partition cleanly.")
        else
            @warn "Consistency check failed ($(length(issues)) issue(s)):\n" * join(issues, "\n")
        end
    end

    return ok
end
"""
    verify_kernelpoints(kp::Kernelpoints, conf; key_dims)

Performs comprehensive validation of a `Kernelpoints` structure for internal consistency.

# Arguments
- `kp::Kernelpoints`: The kernel points structure to verify.
- `conf`: Configuration object (default: `get_empty_config()`).
- `key_dims`: Indices of dimensions used for key computation (from configuration).

# Returns
- `ok::Bool`: `true` if all validation checks pass, `false` if any issue is found.

# Checks
1. All keys in the `keys` field are present in `key_ranges` and vice versa.
2. Key ranges partition the index space [1:n] exactly with no gaps or overlaps.
3. Each datapoint's recomputed key matches the key bucket assigned to its index.
4. No index is assigned to multiple keys.
"""
function verify_kernelpoints(kp::Kernelpoints, conf = get_empty_config();
                              key_dims = get_key_dims(conf))
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

"""
    struct SimMat{T}

A structure storing precomputed similarity matrix features of a single structure for efficient Hamiltonian evaluation.

# Fields
- `feature_vec :: Vector{T}`: Feature vectors per real-space lattice vector `R`, each a tuple `(entries, nnz)`
  where `entries[n] = (sparse_feature_vector, (i_mat, j_mat, key))`.
- `feature_shape :: Tuple{Tuple{Int64, Int64}, Int64}`: Tuple containing ((N_R, Ne), number_of_datapoints).
- `sim_params :: Float64`: Similarity function parameter (e.g., Gaussian width σ).

# Notes
- Feature vectors contain sparse vectors computed as exponential similarities between data points and structure descriptors.
- The feature vectors are filtered by a tolerance to eliminate small contributions.
"""
struct SimMat{T}
    feature_vec :: Vector{T}
    feature_shape :: Tuple{Tuple{Int64, Int64}, Int64}
    sim_params :: Float64
end

"""
    simmat_type(nkeys) -> Type

Returns the concrete `SimMat` type produced by `get_kernel_features` for `nkeys` key dimensions.
"""
function simmat_type(nkeys)
    EntryT = Tuple{SparseVector{Float64, Int64}, Tuple{Int64, Int64, NTuple{nkeys,Int64}}}
    return SimMat{Tuple{Vector{EntryT}, Int64}}
end

"""
    get_empty_simmat(nkeys=1; sim_params=0.0) -> SimMat
    get_empty_simmat(sm::SimMat) -> SimMat

Returns an empty `SimMat` (no `R` blocks, shape `((0, 0), 0)`) with the same type as the output of
`get_kernel_features` for `nkeys` key dimensions, or with the same type and `sim_params` as `sm`.
"""
get_empty_simmat(nkeys = 1; sim_params = 0.0) = get_empty_simmat(simmat_type(nkeys), sim_params)
get_empty_simmat(sm::SimMat) = get_empty_simmat(typeof(sm), sm.sim_params)
get_empty_simmat(::Type{SimMat{T}}, sim_params) where {T} = SimMat(T[], ((0, 0), 0), Float64(sim_params))



"""
    get_kernel_features(structure_descriptors, kp, sim_params, tol; conf, rank, systems, key_dims, label)

Generates precomputed kernel feature vectors based on structure descriptors and sorted kernel points.

# Arguments
- `structure_descriptors`: Vector of structure descriptor matrices (one per structure).
- `kp::Kernelpoints`: Sorted kernel points containing data points and key ranges.
- `sim_params::Float64`: Gaussian width parameter σ for exponential similarity function.
- `tol::Float64`: Tolerance threshold for filtering small feature values (default: 0.1).
- `conf`: Configuration object (default: `get_empty_config()`).
- `rank`: MPI rank for distributed computation and logging (default: 0).
- `systems`: Labels for each structure (default: auto-generated strings).
- `key_dims`: Key dimensions from configuration.
- `label`: Name of the Hamiltonian set (e.g. `"train"`, `"val"`, `"standard"`) shown in the log messages (default: `""`).
- `coverage_file`: File to which the coverage log messages are appended (default: `"ml_coverage.txt"`).

# Returns
- `Vector{SimMat{T}}`: One similarity matrix per structure containing feature vectors, descriptor shape, and parameters.

# Notes
- Computes exponential similarities between each structural descriptor element and all data points in its key bucket.
- Filters features below tolerance to maintain sparsity.
- Provides detailed logging of coverage statistics (number of non-zero features vs. total elements).
"""
function get_kernel_features(structure_descriptors, kp, sim_params, tol = 0.1; conf = get_empty_config(), rank = 0, systems = nothing, key_dims = get_key_dims(conf), hyperopt_iter = get_hyperopt_niter(conf), verbosity = get_verbosity(conf), sim_tol = get_sim_tol(conf), label = "", coverage_file = "ml_coverage.txt")
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
                val_vec[abs.(val_vec) .<= sim_tol] .= 0
                val_vec = sparse(val_vec)
                covered = nnz(val_vec) > 0 ? 1 : 0
                N_test_temp[n] = covered
                Desc_Vec[i][R][1][n] = (val_vec,(i_mat,j_mat, key))
            end
            N_test[i] += sum(N_test_temp)
            N_total[i] += N_total_temp
        end
        if hyperopt_iter < 2 && verbosity > 0
            msg = "Rank $rank$(isempty(label) ? "" : " ($label)"): Finished kernel features for mat $(systems[i]) Nr. ($i / $N_mats) with Ncovered = ( $(N_test[i]) / $(N_total[i]) ) || $(ceil(Int, N_test[i] / N_total[i] * 100 )) %"
            @info msg
            open(coverage_file, "a") do file
                println(file, msg)
            end
        end
    end
    structure_descriptors = nothing
    GC.gc()
    return [SimMat(Desc_Vec[i], (descr_sizes[i], N_dp), sim_params) for i in 1:N_mats]
end

"""
    simmat_filename(rank; prefix="simmat", dir="tmp") -> String

Returns the per-rank HDF5 file path `"\$(dir)/\$(prefix)_rank\$(rank).h5"` used by `write_simmat` and `read_simmat`.
"""
simmat_filename(rank; prefix = "simmat", dir = "tmp") = joinpath(dir, string(prefix, "_rank", rank, ".h5"))

"""
    write_simmat(sms::AbstractVector{<:SimMat}, rank=0; prefix="simmat", dir="tmp", filename=simmat_filename(rank; prefix, dir), append=false)

Writes the vector of `SimMat`s produced by `get_kernel_features` to an HDF5 file (one file per MPI rank, serial HDF5).
The parent directory of `filename` (by default `tmp/`) is created if it does not exist.
With `append=true` the `SimMat`s are appended to an existing file (as structures `N_old+1:N_old+length(sms)`),
so that the features can be computed and written batch by batch. All datasets are chunked and extendible.

# File layout
All nested entries `sms[i].feature_vec[R][1][n]` are flattened in (i, R, n) order:
- `descr_sizes :: Matrix{Int64}` (2 × N_mats): columns `(N_R, Ne)` of `sms[i].feature_shape[1]`.
- `N_dp :: Vector{Int64}`, `sim_params :: Vector{Float64}` (N_mats): remaining fields of each `SimMat`.
- `n_entries :: Vector{Int64}` (Σ N_R): number of entries per (i, R) block.
- `index :: Matrix{Int64}` (2 + nkeys × N_entries): `(i_mat, j_mat, key...)` per entry.
- `sv_len`, `sv_nnz :: Vector{Int64}` (N_entries): length and number of stored values of each sparse vector.
- `sv_nzind :: Vector{Int64}`, `sv_nzval :: Vector{Float64}` (Σ sv_nnz): concatenated sparse vector data.
- `offsets :: Matrix{Int64}` (3 × N_mats + 1): cumulative (block, entry, nonzero) offsets per structure,
  so that a single structure can be read with `read_simmat_index`.
- Attribute `nkeys`.
"""
function write_simmat(sms::AbstractVector{<:SimMat}, rank = 0; prefix = "simmat", dir = "tmp", filename = simmat_filename(rank; prefix = prefix, dir = dir), append = false)
    N_mats = length(sms)
    nkeys = isconcretetype(eltype(sms)) ? _simmat_nkeys(eltype(sms)) : 1

    N_blocks = sum(sm -> length(sm.feature_vec), sms; init = 0)
    N_entries = sum(sm -> sum(b -> length(b[1]), sm.feature_vec; init = 0), sms; init = 0)
    N_nz = sum(sm -> sum(b -> sum(e -> nnz(e[1]), b[1]; init = 0), sm.feature_vec; init = 0), sms; init = 0)

    descr_mat = zeros(Int64, 2, N_mats)
    N_dp = zeros(Int64, N_mats)
    sim_params = zeros(Float64, N_mats)
    n_entries = zeros(Int64, N_blocks)
    index = zeros(Int64, 2 + nkeys, N_entries)
    sv_len = zeros(Int64, N_entries)
    sv_nnz = zeros(Int64, N_entries)
    sv_nzind = zeros(Int64, N_nz)
    sv_nzval = zeros(Float64, N_nz)
    offsets = zeros(Int64, 3, N_mats + 1)

    b = 0; e = 0; z = 0
    for (i, sm) in enumerate(sms)
        offsets[:, i] .= (b, e, z)
        (descr_mat[1, i], descr_mat[2, i]), N_dp[i] = sm.feature_shape
        sim_params[i] = sm.sim_params
        for block in sm.feature_vec
            b += 1
            n_entries[b] = length(block[1])
            for (sv, (i_mat, j_mat, key)) in block[1]
                e += 1
                index[1, e] = i_mat
                index[2, e] = j_mat
                for k in 1:nkeys
                    index[2 + k, e] = key[k]
                end
                sv_len[e] = length(sv)
                nz = nnz(sv)
                sv_nnz[e] = nz
                sv_nzind[z+1:z+nz] .= SparseArrays.nonzeroinds(sv)
                sv_nzval[z+1:z+nz] .= SparseArrays.nonzeros(sv)
                z += nz
            end
        end
    end
    offsets[:, N_mats + 1] .= (b, e, z)

    datasets = ("descr_sizes" => descr_mat, "N_dp" => N_dp, "sim_params" => sim_params, "n_entries" => n_entries,
                "index" => index, "sv_len" => sv_len, "sv_nnz" => sv_nnz, "sv_nzind" => sv_nzind, "sv_nzval" => sv_nzval)

    mkpath(dirname(abspath(filename)))
    if append && isfile(filename)
        h5open(filename, "r+") do file
            nkeys_file = read_attribute(file, "nkeys")
            nkeys_file == nkeys || error("Cannot append SimMats with nkeys = $nkeys to $filename (nkeys = $nkeys_file)")
            # shift the local offsets by the totals already stored in the file
            offsets .+= file["offsets"][:, end]
            for (name, data) in datasets
                _append_dataset!(file[name], data)
            end
            _append_dataset!(file["offsets"], offsets[:, 2:end])
        end
    else
        h5open(filename, "w") do file
            for (name, data) in datasets
                _create_dataset(file, name, data)
            end
            _create_dataset(file, "offsets", offsets)
            attributes(file)["nkeys"] = nkeys
        end
    end
    return filename
end

# Creates a chunked dataset that is extendible along its last dimension and fills it with data.
function _create_dataset(file, name, data::AbstractArray{T}) where {T}
    dims = size(data)
    max_dims = (dims[1:end-1]..., -1)
    chunk = (dims[1:end-1]..., clamp(nextpow(2, max(dims[end], 1)), 64, 2^16))
    dset = create_dataset(file, name, datatype(T), dataspace(dims; max_dims = max_dims); chunk = chunk)
    isempty(data) || write(dset, data)
    return dset
end

# Appends data to an extendible dataset along its last dimension.
function _append_dataset!(dset, data::AbstractArray)
    isempty(data) && return dset
    dims = size(dset)
    n = dims[end]
    HDF5.set_extent_dims(dset, (dims[1:end-1]..., n + size(data)[end]))
    if ndims(data) == 1
        dset[n+1:n+length(data)] = data
    else
        dset[:, n+1:n+size(data, 2)] = data
    end
    return dset
end

_simmat_nkeys(::Type{SimMat{Tuple{Vector{Tuple{SparseVector{Float64, Int64}, Tuple{Int64, Int64, NTuple{N,Int64}}}}, Int64}}}) where {N} = N

"""
    read_simmat(rank=0; prefix="simmat", dir="tmp", filename=simmat_filename(rank; prefix, dir)) -> Vector{SimMat}

Reads the `SimMat`s previously written with `write_simmat` and rebuilds them with the same nested structure
and types as returned by `get_kernel_features`.
"""
function read_simmat(rank = 0; prefix = "simmat", dir = "tmp", filename = simmat_filename(rank; prefix = prefix, dir = dir))
    data = h5open(filename, "r") do file
        (descr_mat = read(file, "descr_sizes"),
         N_dp = read(file, "N_dp"),
         sim_params = read(file, "sim_params"),
         n_entries = read(file, "n_entries"),
         index = read(file, "index"),
         sv_len = read(file, "sv_len"),
         sv_nnz = read(file, "sv_nnz"),
         sv_nzind = read(file, "sv_nzind"),
         sv_nzval = read(file, "sv_nzval"),
         nkeys = read_attribute(file, "nkeys"))
    end
    return _build_simmats(data, Val(Int(data.nkeys)))
end

"""
    read_simmat_index(index, rank=0; prefix="simmat", dir="tmp", filename=simmat_filename(rank; prefix, dir)) -> SimMat

Reads only the `SimMat` of structure `index` from a file written with `write_simmat`, using the stored
`offsets` to load just the corresponding slices of each dataset.
"""
function read_simmat_index(index::Integer, rank = 0; prefix = "simmat", dir = "tmp", filename = simmat_filename(rank; prefix = prefix, dir = dir))
    data = h5open(filename, "r") do file
        N_mats = size(file["descr_sizes"], 2)
        1 <= index <= N_mats || throw(BoundsError("structure index $index not in 1:$N_mats ($filename)"))
        (b0, e0, z0), (b1, e1, z1) = file["offsets"][:, index], file["offsets"][:, index + 1]
        (descr = file["descr_sizes"][:, index],
         N_dp = file["N_dp"][index],
         sim_params = file["sim_params"][index],
         n_entries = _read_range(file["n_entries"], b0+1:b1),
         index = _read_range(file["index"], e0+1:e1),
         sv_len = _read_range(file["sv_len"], e0+1:e1),
         sv_nnz = _read_range(file["sv_nnz"], e0+1:e1),
         sv_nzind = _read_range(file["sv_nzind"], z0+1:z1),
         sv_nzval = _read_range(file["sv_nzval"], z0+1:z1),
         nkeys = read_attribute(file, "nkeys"))
    end
    nkeys = Int(data.nkeys)
    return _build_simmat(simmat_type(nkeys), Val(nkeys), data.descr, data.N_dp, data.sim_params,
                         data.n_entries, data.index, data.sv_len, data.sv_nnz, data.sv_nzind, data.sv_nzval)
end

# Reads the range r along the last dimension of dset (hyperslab), returning an empty array for an empty range.
function _read_range(dset, r)
    if ndims(dset) == 1
        return isempty(r) ? eltype(dset)[] : dset[r]
    else
        return isempty(r) ? zeros(eltype(dset), size(dset, 1), 0) : dset[:, r]
    end
end

function _build_simmats(data, ::Val{nkeys}) where {nkeys}
    SimMatT = simmat_type(nkeys)
    N_mats = size(data.descr_mat, 2)

    b = 0; e = 0; z = 0
    sms = Vector{SimMatT}(undef, N_mats)
    for i in 1:N_mats
        N_R = data.descr_mat[1, i]
        n_ent = sum(@view(data.n_entries[b+1:b+N_R]); init = 0)
        n_nz = sum(@view(data.sv_nnz[e+1:e+n_ent]); init = 0)
        sms[i] = _build_simmat(SimMatT, Val(nkeys), @view(data.descr_mat[:, i]), data.N_dp[i], data.sim_params[i],
                               @view(data.n_entries[b+1:b+N_R]), @view(data.index[:, e+1:e+n_ent]),
                               @view(data.sv_len[e+1:e+n_ent]), @view(data.sv_nnz[e+1:e+n_ent]),
                               @view(data.sv_nzind[z+1:z+n_nz]), @view(data.sv_nzval[z+1:z+n_nz]))
        b += N_R; e += n_ent; z += n_nz
    end
    return sms
end

# Rebuilds a single SimMat from the flattened arrays of one structure (all indexed starting at 1).
function _build_simmat(::Type{SimMatT}, ::Val{nkeys}, descr, N_dp, sim_params, n_entries, index, sv_len, sv_nnz, sv_nzind, sv_nzval) where {SimMatT, nkeys}
    BlockT = eltype(fieldtype(SimMatT, :feature_vec))
    EntryT = eltype(fieldtype(BlockT, 1))
    N_R, Ne = Int64(descr[1]), Int64(descr[2])

    e = 0; z = 0
    blocks = Vector{BlockT}(undef, N_R)
    for R in 1:N_R
        n = n_entries[R]
        entries = Vector{EntryT}(undef, n)
        for m in 1:n
            e += 1
            nz = sv_nnz[e]
            sv = SparseVector{Float64, Int64}(sv_len[e], sv_nzind[z+1:z+nz], sv_nzval[z+1:z+nz])
            z += nz
            key = ntuple(k -> Int64(index[2 + k, e]), nkeys)
            entries[m] = (sv, (Int64(index[1, e]), Int64(index[2, e]), key))
        end
        blocks[R] = (entries, n)
    end
    return SimMat(blocks, ((N_R, Ne), Int64(N_dp)), Float64(sim_params))
end

"""
    exp_sim2(x1::SVector{9,Float64}, x2::SVector{9,Float64}; σ, dim_weights)

Computes the exponential similarity (Gaussian RBF kernel) between two 9-dimensional vectors.

# Arguments
- `x1::SVector{9,Float64}`: First input vector.
- `x2::SVector{9,Float64}`: Second input vector.
- `σ`: Gaussian width parameter (default: `√0.05`).
- `dim_weights`: Per-dimension scaling weights (default: `ones(9)`).

# Returns
- `similarity::Float64`: Exponential similarity value in range [0, 1].

# Formula
Computes: `exp(-||x1 ⊙ w - x2 ⊙ w||² / (2σ²))` where ⊙ denotes element-wise multiplication.
"""
exp_sim2(x1::SVector{9,Float64}, x2::SVector{9,Float64}; σ=√0.05, dim_weights = ones(9)) =
    exp(-normdiff(x1.*dim_weights, x2.*dim_weights)^2 / (2σ^2))

"""
    exp_sim_all(x1, x2::SVector{9,Float64}; σ, dim_weights)

Broadcasts the exponential similarity function over a vector of points x1 against a single point x2.

# Arguments
- `x1`: Vector of 9-dimensional points.
- `x2::SVector{9,Float64}`: Reference point for comparison.
- `σ`: Gaussian width parameter (default: `√0.05`).
- `dim_weights`: Per-dimension scaling weights (default: `ones(9)`).

# Returns
- `similarities::Vector`: Exponential similarities between each point in x1 and x2.

# Notes
- Vectorized version of `exp_sim2` for efficient computation against a reference point.
"""
exp_sim_all(x1, x2::SVector{9,Float64}; σ=√0.05, dim_weights = ones(9)) = exp_sim2.(x1, Ref(x2); σ=σ, dim_weights=dim_weights)