# All Hamiltonian models are linear in their parameters. For one structure, the nonzero real-space matrix
# elements ("entries" (R, i, j)) of a model can thus be written as hr = A * params. The device code path
# precomputes these linear maps once and evaluates Hamiltonians and gradients with device kernels.

"""
    SparseTerms(keys, cols, vals, nparams)

Host representation of a linear model contribution `Hr[R][i, j] += vals[n] * params[cols[n]]` with `keys[n] = (R, i, j)`.
"""
struct SparseTerms{T}
    keys :: Vector{NTuple{3, Int64}}
    cols :: Vector{Int64}
    vals :: Vector{T}
    nparams :: Int64
end

SparseTerms(T, nparams) = SparseTerms(NTuple{3, Int64}[], Int64[], T[], nparams)

"""
    GaussianKernelTerms(keys, X, D, σ, weights, ptr)

Host representation of the Gaussian kernel model
`Hr[R][i, j] = Σ_{t ∈ ptr[u]:ptr[u+1]-1} weights[t] Σ_n params[n] * exp(-|X[:, t] - D[:, n]|² / 2σ²)` with
`keys[u] = (R, i, j)`: every matrix element `u` has the descriptors `X[:, t]` of its variants (enforce_symmetry_group; one
descriptor with weight 1 otherwise).
"""
struct GaussianKernelTerms
    keys :: Vector{NTuple{3, Int64}}
    X :: Matrix{Float64}
    D :: Matrix{Float64}
    σ :: Float64
    weights :: Vector{Float64}
    ptr :: Vector{Int64}
end

"""
    spin_keys(R, i, j, soc)

Return the entries of `kron(H, I_spin)` that the element `H[i, j]` contributes to (see `apply_spin_basis`).
"""
spin_keys(R, i, j, soc) = soc ? ((R, 2i-1, 2j-1), (R, 2i, 2j)) : ((R, i, j),)

function push_term!(terms::SparseTerms, R, i, j, col, val; soc=false)
    for key in spin_keys(R, i, j, soc)
        push!(terms.keys, key); push!(terms.cols, col); push!(terms.vals, val)
    end
end

"""
    linear_terms(model, index; soc=false)

Return the linear terms (`SparseTerms` or `GaussianKernelTerms`) of `model` for the structure `index`, such that the sum of
all terms reproduces `get_hr(model, mode, index, apply_soc=soc)`.
"""
function linear_terms(model::TBModel, index; soc=false)
    h = model.hs[index]
    global_params = model.params_per_strc[index]
    terms = SparseTerms(Float64, length(model.params))
    for R in axes(h, 2), v in axes(h, 1)
        for (i, j, val) in zip(findnz(h[v, R])...)
            push_term!(terms, R, i, j, global_params[v], val, soc=soc)
        end
    end
    return terms
end

function linear_terms(kernel::HamiltonianKernel, index; soc=false)
    h_env = kernel.structure_descriptors[index]
    symmetry = isnothing(kernel.symmetry) ? nothing : kernel.symmetry[index]
    Ndim = length(first(kernel.data_points))
    keys = NTuple{3, Int64}[]
    columns = Vector{Float64}[]; weights = Float64[]; ptr = [1]
    for R in eachindex(h_env)
        # Variants of every stored descriptor (enforce_symmetry_group), by its index in the order of findnz
        variants = [Tuple{SVector{8, Float64}, Float64}[] for _ in 1:nnz(h_env[R])]
        isnothing(symmetry) || foreach(((k, x, w),) -> push!(variants[k], (x, w)), symmetry.extra[R])
        for (k, (i, j, hin)) in enumerate(zip(findnz(h_env[R])...))
            w0 = isnothing(symmetry) ? 1.0 : symmetry.base_weights[R][k]
            for key in spin_keys(R, i, j, soc)
                push!(keys, key)
                push!(columns, Vector{Float64}(hin)); push!(weights, w0)
                for (x, w) in variants[k]
                    push!(columns, Vector{Float64}(x)); push!(weights, w)
                end
                push!(ptr, length(weights) + 1)
            end
        end
    end
    X = isempty(columns) ? zeros(Ndim, 0) : reduce(hcat, columns)
    D = reduce(hcat, Vector{Float64}.(kernel.data_points))
    return GaussianKernelTerms(keys, X, D, kernel.sim_params, weights, ptr)
end

function linear_terms(kernel::HamiltonianKernelPrecalced, index; soc=false)
    desc_vec = kernel.sm.feature_vec[index]
    key_ranges = kernel.kp.key_ranges
    terms = SparseTerms(Float64, length(kernel.params))
    for R in eachindex(desc_vec)
        features, nnz_ham = desc_vec[R]
        for m in 1:nnz_ham
            feature, (i, j, key) = features[m]
            param_range = key_ranges[key]
            for (n, val) in zip(findnz(feature)...)
                push_term!(terms, R, i, j, param_range[n], val, soc=soc)
            end
        end
    end
    return terms
end

function linear_terms(soc_model::SOCModel, index; soc=true)
    R0 = soc_model.Rs_info[2, index]
    terms = SparseTerms(ComplexF64, length(soc_model.params))
    offset = 0
    for type in soc_model.types_per_strc[index]
        M = soc_model.matrices[type]
        p = findfirst(label -> label == type, soc_model.param_labels)
        for b in axes(M, 2), a in axes(M, 1)
            if M[a, b] ≠ 0
                push_term!(terms, R0, offset + a, offset + b, p, M[a, b])
            end
        end
        offset += size(M, 1)
    end
    return terms
end

function linear_terms(ewald::EwaldOnsites, index; soc=false)
    R0 = ewald.Rs_info[2, index]
    Norb = soc ? 2 .* ewald.norb_per_strc[index] : ewald.norb_per_strc[index]
    terms = SparseTerms(Float64, length(ewald.params))
    offset = 0
    for (iion, ϕ) in enumerate(ewald.potentials[index])
        p = findfirst(label -> label == ewald.types_per_strc[index][iion], ewald.param_labels)
        for n in 1:Norb[iion]
            push_term!(terms, R0, offset + n, offset + n, p, -ϕ)
        end
        offset += Norb[iion]
    end
    return terms
end

"""
    DeviceSparseBlock

Device representation of `SparseTerms` of model `model_index`: the matrix `A` (entries × parameters) in CSR format
and its transpose `At` (parameters × entries) in CSR format.
"""
struct DeviceSparseBlock{IV, V}
    model_index :: Int64
    nparams :: Int64
    rowptr :: IV
    colval :: IV
    nzval :: V
    t_rowptr :: IV
    t_colval :: IV
    t_nzval :: V
end

"""
    DeviceKernelBlock

Device representation of `GaussianKernelTerms` of model `model_index`. `rows[u]` is the entry of matrix element `u`,
whose descriptors are the columns `ptr[u]:ptr[u+1]-1` of `X` with the weights `w`.
"""
struct DeviceKernelBlock{IV, M, V}
    model_index :: Int64
    nparams :: Int64
    rows :: IV
    X :: M
    D :: M
    γ :: Float64
    w :: V
    ptr :: IV
end

"""
    DeviceLinearMap

Precomputed, device-resident description of the real- and k-space Hamiltonians of one structure.

# Fields
- `Nε`, `NR`: Matrix dimension and number of lattice vectors.
- `I`, `J`, `Ridx`: Row, column and lattice-vector index of each entry of the real-space Hamiltonian.
- `blocks`: The linear maps from the model parameters to the entries.
- `hk_rowptr`, `hk_colval`, `hk_rowval`: Sparsity pattern of the k-space Hamiltonian (CSR, always including the diagonal).
- `grp_ptr`, `grp_ent`: For each nonzero of the k-space Hamiltonian, the entries that contribute to it (CSR layout).
- `diag_pos`: Position of each diagonal element in the nonzeros of the k-space Hamiltonian.
"""
struct DeviceLinearMap{IV, B}
    Nε :: Int64
    NR :: Int64
    I :: IV
    J :: IV
    Ridx :: IV
    blocks :: B
    hk_rowptr :: IV
    hk_colval :: IV
    hk_rowval :: IV
    grp_ptr :: IV
    grp_ent :: IV
    diag_pos :: IV
end

nentries(lmap::DeviceLinearMap) = length(lmap.I)
nnz_hk(lmap::DeviceLinearMap) = length(lmap.hk_colval)

# Ptr array of CSR/CSC format from (sorted) row indices
function counts_to_ptr(rows, n)
    ptr = zeros(Int64, n + 1); ptr[1] = 1
    for r in rows; ptr[r+1] += 1; end
    return cumsum(ptr)
end

encode_entry(key, Nε) = ((key[1] - 1) * Nε + (key[2] - 1)) * Nε + (key[3] - 1)

"""
    host_linear_map(models, index, Nε, NR; soc=false)

The arrays of the `DeviceLinearMap` of structure `index` on the host (see `DeviceLinearMap`).
"""
function host_linear_map(models, index, Nε, NR; soc=false)
    all_terms = [(m, linear_terms(model, index, soc=soc)) for (m, model) in enumerate(models)]

    # Union of all entries, sorted by (R, i, j)
    codes = unique!(sort!(reduce(vcat, [encode_entry.(terms.keys, Nε) for (_, terms) in all_terms], init=Int64[])))
    Nent = length(codes)
    Ridx = codes .÷ Nε^2 .+ 1
    I = mod.(codes .÷ Nε, Nε) .+ 1
    J = mod.(codes, Nε) .+ 1
    entry_index(key) = searchsortedfirst(codes, encode_entry(key, Nε))

    blocks = map(all_terms) do (m, terms)
        rows = tmap(entry_index, terms.keys)
        if terms isa SparseTerms
            A = sparse(rows, terms.cols, terms.vals, Nent, terms.nparams)
            At = copy(transpose(A))          # no duplicate (entry, parameter) pairs, so this equals sparse(cols, rows, vals)
            (m=m, nparams=terms.nparams, A=A, At=At)
        else
            allunique(rows) || error("Kernel model contains duplicate matrix elements.")
            (m=m, nparams=size(terms.D, 2), rows=rows, X=terms.X, D=terms.D, γ=1 / (2 * terms.σ^2), w=terms.weights, ptr=terms.ptr)
        end
    end

    # Sparsity pattern of Hk in CSR format (row-major codes), always including the diagonal
    ij = (I .- 1) .* Nε .+ (J .- 1)
    diag_codes = [(i - 1) * Nε + (i - 1) for i in 1:Nε]
    hk_codes = unique!(sort!(vcat(ij, diag_codes)))
    hk_rowval = hk_codes .÷ Nε .+ 1
    hk_colval = mod.(hk_codes, Nε) .+ 1
    hk_rowptr = counts_to_ptr(hk_rowval, Nε)

    # Group entries by their position in Hk (the position is monotonic in ij, so sorting by ij groups them), and the
    # positions of the entries and of the diagonal by walking through the sorted codes
    grp_ent = sortperm(ij)
    hk_pos = Vector{Int64}(undef, Nent)
    pos = 1
    for e in grp_ent
        while hk_codes[pos] < ij[e]; pos += 1; end
        hk_pos[e] = pos
    end
    diag_pos = Vector{Int64}(undef, Nε)
    pos = 1
    for i in 1:Nε
        while hk_codes[pos] < diag_codes[i]; pos += 1; end
        diag_pos[i] = pos
    end
    grp_ptr = counts_to_ptr(hk_pos, length(hk_codes))
    return (Nε=Nε, NR=NR, I=I, J=J, Ridx=Ridx, blocks=blocks, hk_rowptr=hk_rowptr, hk_colval=hk_colval, hk_rowval=hk_rowval,
            grp_ptr=grp_ptr, grp_ent=grp_ent, diag_pos=diag_pos)
end

"""
    DeviceLinearMap(host, device)
    DeviceLinearMap(models, index, Nε, NR, device; soc=false)

Move the host arrays of a linear map (`host_linear_map`) to `device`, or build the `DeviceLinearMap` of structure `index`
for the models of an `EffectiveHamiltonian`.
"""
function DeviceLinearMap(host::NamedTuple, device)
    Ti = Int32
    dev(x) = to_device(device, Ti.(x))
    blocks = map(host.blocks) do b
        if haskey(b, :A)
            T = eltype(b.A.nzval)
            # The CSC arrays of At (A) are the CSR arrays of A (At)
            DeviceSparseBlock(b.m, b.nparams, dev(b.At.colptr), dev(b.At.rowval), to_device(device, T.(b.At.nzval)),
                dev(b.A.colptr), dev(b.A.rowval), to_device(device, T.(b.A.nzval)))
        else
            DeviceKernelBlock(b.m, b.nparams, dev(b.rows), to_device(device, b.X), to_device(device, b.D), b.γ,
                              to_device(device, b.w), dev(b.ptr))
        end
    end
    return DeviceLinearMap(host.Nε, host.NR, dev(host.I), dev(host.J), dev(host.Ridx), blocks, dev(host.hk_rowptr),
        dev(host.hk_colval), dev(host.hk_rowval), dev(host.grp_ptr), dev(host.grp_ent), dev(host.diag_pos))
end

DeviceLinearMap(models, index, Nε, NR, device; soc=false) = DeviceLinearMap(host_linear_map(models, index, Nε, NR, soc=soc), device)
