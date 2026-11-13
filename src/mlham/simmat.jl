
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
    struct SimMatBlock

Precomputed kernel features of the matrix elements of one real-space block `R` of a structure, in a flat (CSR-like) layout.
Entry `m` is the matrix element `(is[m], js[m])`; its feature vector has the stored values `val[q]` at the (global)
parameter indices `idx[q]` for `q ∈ ptr[m]:ptr[m+1]-1`, such that `H[is[m], js[m]] = Σ_q val[q] * params[idx[q]]`.

Compared to one `SparseVector` per entry, this layout needs no per-entry heap allocations and stores 8 instead of 16 bytes
per nonzero (`Int32` index, `Float32` value; the similarities lie in `(sim_tol, 1]`, so `Float32` is accurate to ~1e-7).
"""
struct SimMatBlock
    is :: Vector{Int32}
    js :: Vector{Int32}
    ptr :: Vector{Int64}
    idx :: Vector{Int32}
    val :: Vector{Float32}
end

SimMatBlock() = SimMatBlock(Int32[], Int32[], [1], Int32[], Float32[])

"""
    nentries(block::SimMatBlock)

Number of matrix elements (entries) of a `SimMatBlock`.
"""
nentries(block::SimMatBlock) = length(block.is)
SparseArrays.nnz(block::SimMatBlock) = length(block.idx)

"""
    struct SimMat

Precomputed similarity matrix features of a single structure for efficient Hamiltonian evaluation.

# Fields
- `blocks :: Vector{SimMatBlock}`: The features of the matrix elements per real-space lattice vector `R`.
- `feature_shape :: Tuple{Tuple{Int64, Int64}, Int64}`: `((N_R, Ne), number_of_datapoints)`.
- `sim_params :: Float64`: Similarity function parameter (Gaussian width σ).
"""
struct SimMat
    blocks :: Vector{SimMatBlock}
    feature_shape :: Tuple{Tuple{Int64, Int64}, Int64}
    sim_params :: Float64
end

SparseArrays.nnz(sm::SimMat) = sum(nnz, sm.blocks; init = 0)
nentries(sm::SimMat) = sum(nentries, sm.blocks; init = 0)

"""
    simmat_bytes(sm::SimMat) -> Int64

Approximate memory of the stored features of a `SimMat` in bytes (16 per entry and 8 per nonzero).
"""
simmat_bytes(sm::SimMat) = 16 * nentries(sm) + 8 * nnz(sm)

"""
    get_empty_simmat(; sim_params=0.0) -> SimMat
    get_empty_simmat(sm::SimMat) -> SimMat

Returns an empty `SimMat` (no `R` blocks, shape `((0, 0), 0)`), with the `sim_params` of `sm` in the second form.
"""
get_empty_simmat(; sim_params = 0.0) = SimMat(SimMatBlock[], ((0, 0), 0), Float64(sim_params))
get_empty_simmat(sm::SimMat) = get_empty_simmat(sim_params = sm.sim_params)

"""
    get_kernel_features(structure_descriptors, kp, sim_params, tol; conf, rank, systems, key_dims, label)

Generates precomputed kernel feature vectors based on structure descriptors and sorted kernel points.

# Arguments
- `structure_descriptors`: Vector of structure descriptor matrices (one per structure).
- `kp::Kernelpoints`: Sorted kernel points containing data points and key ranges.
- `sim_params::Float64`: Gaussian width parameter σ for exponential similarity function.
- `tol::Float64`: Unused (kept for the call signature); the cutoff is `sim_tol`.
- `conf`: Configuration object (default: `get_empty_config()`).
- `rank`: MPI rank for distributed computation and logging (default: 0).
- `systems`: Labels for each structure (default: auto-generated strings).
- `key_dims`: Key dimensions from configuration.
- `label`: Name of the Hamiltonian set (e.g. `"train"`, `"val"`, `"standard"`) shown in the log messages (default: `""`).
- `coverage_file`: File to which the coverage log messages are appended (default: `"ml_coverage.txt"`).

# Returns
- `Vector{SimMat}`: One similarity matrix per structure.

# Notes
- Computes exponential similarities between each structural descriptor element and the data points in its key bucket
  within the cutoff radius, keeping the values `> sim_tol`.
- Entries whose key has no data points get no features (they contribute zero).
- Logs per structure the coverage (entries with at least one feature), the mean number of nonzeros per entry and the
  memory of the features.
"""
function get_kernel_features(structure_descriptors, kp, sim_params, tol = 0.1; conf = get_empty_config(), rank = 0, systems = nothing, key_dims = get_key_dims(conf), hyperopt_iter = get_hyperopt_niter(conf), verbosity = get_verbosity(conf), sim_tol = get_sim_tol(conf), label = "", coverage_file = "ml_coverage.txt")
    data_points = kp.datapoints
    key_ranges = kp.key_ranges

    nkeys = max(length(key_dims), 1)
    keyfun = if isempty(key_dims)
        hin -> (0,)
    else
        hin -> ntuple(j -> Int64(hin[key_dims[j]]), nkeys)
    end

    N_mats = length(structure_descriptors)
    systems = systems === nothing ? [string("system_", i) for i in 1:N_mats] : systems
    N_dp = length(data_points)
    bucket_keys = collect(keys(key_ranges))
    bucket_trees = Dict(zip(bucket_keys, tmap(key -> KeyBucketTree(@view(data_points[key_ranges[key]])), bucket_keys)))
    radius = similarity_radius(sim_params, sim_tol)

    sms = Vector{SimMat}(undef, N_mats)
    for i in 1:N_mats
        N_R = length(structure_descriptors[i])
        Ne = size(structure_descriptors[i][1], 1)
        blocks = Vector{SimMatBlock}(undef, N_R)
        N_covered = 0
        for R in 1:N_R
            is, js, hins = findnz(structure_descriptors[i][R])
            # Every task computes the features of a contiguous range of entries into its own flat buffers
            ranges = collect(chunks(eachindex(hins), n = clamp(cld(length(hins), 256), 1, Threads.nthreads())))
            parts = tmap(ranges) do entry_range
                counts = Vector{Int64}(undef, length(entry_range))
                idx = Int32[]; val = Float32[]
                for (t, n) in enumerate(entry_range)
                    hin = hins[n]
                    key = keyfun(hin)
                    nz_before = length(idx)
                    if haskey(bucket_trees, key)
                        kernel_feature!(idx, val, bucket_trees[key], hin, sim_params, sim_tol, radius, first(key_ranges[key]) - 1)
                    end
                    counts[t] = length(idx) - nz_before
                end
                (counts, idx, val)
            end
            counts = reduce(vcat, (p[1] for p in parts); init = Int64[])
            ptr = Vector{Int64}(undef, length(counts) + 1)
            ptr[1] = 1
            for m in eachindex(counts); ptr[m+1] = ptr[m] + counts[m]; end
            blocks[R] = SimMatBlock(Int32.(is), Int32.(js), ptr,
                                    reduce(vcat, (p[2] for p in parts); init = Int32[]),
                                    reduce(vcat, (p[3] for p in parts); init = Float32[]))
            N_covered += count(>(0), counts)
        end
        sms[i] = SimMat(blocks, ((N_R, Ne), N_dp), sim_params)
        if hyperopt_iter < 2 && verbosity > 0
            N_total = nentries(sms[i])
            msg = "Rank $rank$(isempty(label) ? "" : " ($label)"): Finished kernel features for mat $(systems[i]) Nr. ($i / $N_mats) " *
                  "with Ncovered = ( $N_covered / $N_total ) || $(N_total > 0 ? ceil(Int, N_covered / N_total * 100) : 0) % " *
                  "| nnz/entry = $(round(nnz(sms[i]) / max(N_total, 1), digits=1)) | $(round(simmat_bytes(sms[i]) / 2^20, digits=1)) MB"
            @info msg
            open(coverage_file, "a") do file
                println(file, msg)
            end
        end
    end
    return sms
end

"""
    simmat_filename(rank; prefix="simmat", dir="tmp") -> String

Returns the per-rank HDF5 file path `"\$(dir)/\$(prefix)_rank\$(rank).h5"` used by `write_simmat` and `read_simmat`.
"""
simmat_filename(rank; prefix = "simmat", dir = "tmp") = joinpath(dir, string(prefix, "_rank", rank, ".h5"))

"""
    write_simmat(sms::AbstractVector{SimMat}, rank=0; prefix="simmat", dir="tmp", filename=simmat_filename(rank; prefix, dir), append=false)

Writes the vector of `SimMat`s produced by `get_kernel_features` to an HDF5 file (one file per MPI rank, serial HDF5).
The parent directory of `filename` (by default `tmp/`) is created if it does not exist.
With `append=true` the `SimMat`s are appended to an existing file (as structures `N_old+1:N_old+length(sms)`),
so that the features can be computed and written batch by batch. All datasets are chunked and extendible.

# File layout
All blocks and entries are flattened in (i, R, m) order:
- `descr_sizes :: Matrix{Int64}` (2 × N_mats): columns `(N_R, Ne)` of `sms[i].feature_shape[1]`.
- `N_dp :: Vector{Int64}`, `sim_params :: Vector{Float64}` (N_mats): remaining fields of each `SimMat`.
- `n_entries :: Vector{Int64}` (Σ N_R): number of entries per (i, R) block.
- `ij :: Matrix{Int32}` (2 × N_entries): matrix element `(i, j)` of each entry.
- `entry_nnz :: Vector{Int64}` (N_entries): number of stored features of each entry.
- `idx :: Vector{Int32}`, `val :: Vector{Float32}` (Σ entry_nnz): parameter indices and values of the features.
- `offsets :: Matrix{Int64}` (3 × N_mats + 1): cumulative (block, entry, nonzero) offsets per structure,
  so that a single structure can be read with `read_simmat_index`.
"""
function write_simmat(sms::AbstractVector{SimMat}, rank = 0; prefix = "simmat", dir = "tmp", filename = simmat_filename(rank; prefix = prefix, dir = dir), append = false)
    N_mats = length(sms)
    all_blocks = [block for sm in sms for block in sm.blocks]

    descr_mat = zeros(Int64, 2, N_mats)
    N_dp = zeros(Int64, N_mats)
    sim_params = zeros(Float64, N_mats)
    offsets = zeros(Int64, 3, N_mats + 1)
    b = 0; e = 0; z = 0
    for (i, sm) in enumerate(sms)
        offsets[:, i] .= (b, e, z)
        (descr_mat[1, i], descr_mat[2, i]), N_dp[i] = sm.feature_shape
        sim_params[i] = sm.sim_params
        b += length(sm.blocks); e += nentries(sm); z += nnz(sm)
    end
    offsets[:, N_mats + 1] .= (b, e, z)

    n_entries = Int64[nentries(block) for block in all_blocks]
    ij = permutedims(hcat(reduce(vcat, (block.is for block in all_blocks); init = Int32[]),
                          reduce(vcat, (block.js for block in all_blocks); init = Int32[])))
    entry_nnz = reduce(vcat, (diff(block.ptr) for block in all_blocks); init = Int64[])
    idx = reduce(vcat, (block.idx for block in all_blocks); init = Int32[])
    val = reduce(vcat, (block.val for block in all_blocks); init = Float32[])

    datasets = ("descr_sizes" => descr_mat, "N_dp" => N_dp, "sim_params" => sim_params, "n_entries" => n_entries,
                "ij" => ij, "entry_nnz" => entry_nnz, "idx" => idx, "val" => val)

    mkpath(dirname(abspath(filename)))
    if append && isfile(filename)
        h5open(filename, "r+") do file
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

"""
    read_simmat(rank=0; prefix="simmat", dir="tmp", filename=simmat_filename(rank; prefix, dir)) -> Vector{SimMat}

Reads all `SimMat`s previously written with `write_simmat`.
"""
function read_simmat(rank = 0; prefix = "simmat", dir = "tmp", filename = simmat_filename(rank; prefix = prefix, dir = dir))
    N_mats = h5open(file -> size(file["descr_sizes"], 2), filename, "r")
    return [read_simmat_index(i; filename = filename) for i in 1:N_mats]
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
         ij = _read_range(file["ij"], e0+1:e1),
         entry_nnz = _read_range(file["entry_nnz"], e0+1:e1),
         idx = _read_range(file["idx"], z0+1:z1),
         val = _read_range(file["val"], z0+1:z1))
    end
    N_R, Ne = Int64(data.descr[1]), Int64(data.descr[2])
    blocks = Vector{SimMatBlock}(undef, N_R)
    e = 0; z = 0
    for R in 1:N_R
        n = data.n_entries[R]
        counts = @view data.entry_nnz[e+1:e+n]
        ptr = Vector{Int64}(undef, n + 1)
        ptr[1] = 1
        for m in 1:n; ptr[m+1] = ptr[m] + counts[m]; end
        nz = ptr[end] - 1
        blocks[R] = SimMatBlock(data.ij[1, e+1:e+n], data.ij[2, e+1:e+n], ptr, data.idx[z+1:z+nz], data.val[z+1:z+nz])
        e += n; z += nz
    end
    return SimMat(blocks, ((N_R, Ne), Int64(data.N_dp)), Float64(data.sim_params))
end

# Reads the range r along the last dimension of dset (hyperslab), returning an empty array for an empty range.
function _read_range(dset, r)
    if ndims(dset) == 1
        return isempty(r) ? eltype(dset)[] : dset[r]
    else
        return isempty(r) ? zeros(eltype(dset), size(dset, 1), 0) : dset[:, r]
    end
end

"""
    exp_sim(x1, x2; σ)

Computes the exponential similarity (Gaussian RBF kernel) `exp(-||x1 - x2||² / (2σ²))` between two vectors (default `σ = √0.05`).
"""
exp_sim(x1, x2; σ=√0.05) = exp(-normdiff(x1, x2)^2 / (2σ^2))

"""
    similarity_radius(σ, sim_tol) -> Float64

Distance beyond which the similarity `exp_sim(x1, x2; σ)` is at most `sim_tol`, i.e. `σ √(-2 ln sim_tol)` (with a small safety margin).
"""
function similarity_radius(σ, sim_tol)
    sim_tol <= 0 && return Inf
    sim_tol >= 1 && return 0.0
    return σ * sqrt(-2log(sim_tol)) * (1 + 1e-8)
end

"""
    KeyBucketTree(points; leaf_size=16)

k-d tree of the data points of one key bucket, used to find the data points within a radius of a descriptor.
Every node covers the points `points[lo[node]:hi[node]]` (stored in tree order, `perm` maps them to their index in the bucket).
Inner nodes split their points at the median `split_val` along the dimension `split_dim` with the largest spread into the
children `left` and `right`; leaves have `split_dim = 0`.
"""
struct KeyBucketTree{P}
    points :: Vector{P}
    perm :: Vector{Int64}
    split_dim :: Vector{Int64}
    split_val :: Vector{Float64}
    lo :: Vector{Int64}
    hi :: Vector{Int64}
    left :: Vector{Int64}
    right :: Vector{Int64}
end

function KeyBucketTree(points::AbstractVector{P}; leaf_size=16) where {P}
    perm = collect(eachindex(points))
    tree = KeyBucketTree{P}(P[], perm, Int64[], Float64[], Int64[], Int64[], Int64[], Int64[])
    build_node!(tree, points, 1, length(perm), leaf_size)
    append!(tree.points, points[perm])
    return tree
end

function build_node!(tree, points, lo, hi, leaf_size)
    node = length(tree.lo) + 1
    push!(tree.split_dim, 0); push!(tree.split_val, 0.0); push!(tree.lo, lo); push!(tree.hi, hi); push!(tree.left, 0); push!(tree.right, 0)
    hi - lo + 1 <= leaf_size && return node
    inds = view(tree.perm, lo:hi)
    spread(d) = (coords = (points[n][d] for n in inds); maximum(coords) - minimum(coords))
    dim = argmax(spread, 1:length(first(points)))
    spread(dim) > 0 || return node
    inds .= inds[sortperm([points[n][dim] for n in inds])]
    mid = (lo + hi) ÷ 2
    tree.split_dim[node] = dim
    tree.split_val[node] = points[tree.perm[mid]][dim]
    tree.left[node] = build_node!(tree, points, lo, mid, leaf_size)
    tree.right[node] = build_node!(tree, points, mid + 1, hi, leaf_size)
    return node
end

"""
    kernel_feature!(idx, val, tree, x, σ, sim_tol, radius, offset) -> Int

Appends the similarities `exp_sim(point, x; σ)` of the descriptor `x` to the data points of a key bucket that are
`> sim_tol` to `val`, and their parameter indices (`offset` + index in the bucket) to `idx`, sorted by index. Only the
points within `radius` of `x` are evaluated (see `similarity_radius`). Returns the number of appended values.
"""
function kernel_feature!(idx, val, tree::KeyBucketTree, x, σ, sim_tol, radius, offset)
    radius2 = radius^2
    n0 = length(idx)
    stack = [1]
    @inbounds while !isempty(stack)
        node = pop!(stack)
        dim = tree.split_dim[node]
        if dim == 0
            for k in tree.lo[node]:tree.hi[node]
                point = tree.points[k]
                # Cheap rejection of the points outside the radius before the exponential is evaluated
                sum(abs2, point - x) > radius2 && continue
                s = exp_sim(point, x; σ=σ)
                if abs(s) > sim_tol && s ≠ 0
                    push!(idx, Int32(offset + tree.perm[k]))
                    push!(val, Float32(s))
                end
            end
        else
            x[dim] - radius <= tree.split_val[node] && push!(stack, tree.left[node])
            x[dim] + radius >= tree.split_val[node] && push!(stack, tree.right[node])
        end
    end
    # Sort the new values by parameter index (memory locality in `get_hr` and the gradient)
    if length(idx) - n0 > 1
        p = sortperm(@view(idx[n0+1:end]))
        idx[n0+1:end] = idx[n0 .+ p]
        val[n0+1:end] = val[n0 .+ p]
    end
    return length(idx) - n0
end