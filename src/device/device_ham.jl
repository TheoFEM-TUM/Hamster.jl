"""
    DeviceContext

Device-specific data of an `EffectiveHamiltonian`: the device, the `DeviceLinearMap` of each structure, the settings of the
sparse eigensolver used during training, and caches of the device copies of the phase factors and model parameters
(host-device transfers have a large latency compared to the kernels for small structures).
"""
struct DeviceContext{D, L}
    device :: D
    maps :: Vector{L}
    neig :: Int64
    target :: Float64
    method :: String
    phase_cache :: Dict{Tuple{Int64, UInt64}, Any}
    param_cache :: Dict{Int64, Tuple{Vector{Float64}, Any}}
end

function DeviceContext(device, models, strcs, bases, conf=get_empty_config(); soc=get_soc(conf))
    # The host arrays of the structures are built in parallel, the transfers to the device one after another
    hosts = tmap(eachindex(strcs)) do index
        Nε = soc ? 2 * length(bases[index]) : length(bases[index])
        host_linear_map(models, index, Nε, size(strcs[index].Rs, 2), soc=soc)
    end
    maps = [DeviceLinearMap(host, device) for host in hosts]
    return DeviceContext(device, maps, get_neig(conf), get_eig_target(conf), get_diag_method(conf),
        Dict{Tuple{Int64, UInt64}, Any}(), Dict{Int64, Tuple{Vector{Float64}, Any}}())
end

"""
    DeviceHk

A batch of k-space Hamiltonians with a common sparsity pattern (CSR) that lives on a device.
`nzval[:, k]` contains the nonzeros of the Hamiltonian at the `k`-th k-point.
"""
struct DeviceHk{IV, M, SD}
    Nε :: Int64
    rowptr :: IV
    colval :: IV
    rowval :: IV
    diag_pos :: IV
    nzval :: M
    sp_diag :: SD
end

nkpoints(Hk::DeviceHk) = size(Hk.nzval, 2)
hamiltonian_dim(Hk::DeviceHk) = Hk.Nε
hamiltonian_dim(Hk::AbstractVector{<:AbstractMatrix}) = size(Hk[1], 1)

"""
    get_device_hr(lmap, params)

Evaluate the nonzero entries of the real-space Hamiltonian on the device. `params[m]` are the device parameters of model `m`.
"""
function get_device_hr(lmap::DeviceLinearMap, params)
    hr = fill!(similar(lmap.I, ComplexF64, nentries(lmap)), 0)
    for block in lmap.blocks
        accumulate_hr!(hr, block, params[block.model_index])
    end
    return hr
end

function accumulate_hr!(hr, block::DeviceSparseBlock, p)
    launch_kernel!(csr_matvec_add_kernel!, hr, length(hr), block.rowptr, block.colval, block.nzval, p)
end

function accumulate_hr!(hr, block::DeviceKernelBlock, p)
    launch_kernel!(gaussian_kernel_hr_kernel!, hr, length(block.rows), block.rows, block.X, block.D, p, block.γ)
end

"""
    parameter_gradient(block, G)

Gradient `real(transpose(A) * G)` of the loss w.r.t. the parameters of a block, given the gradient `G` w.r.t. the entries.
"""
function parameter_gradient(block::DeviceSparseBlock, G)
    g = similar(G, Float64, block.nparams)
    launch_kernel!(csr_real_matvec_kernel!, g, block.nparams, block.t_rowptr, block.t_colval, block.t_nzval, G)
end

function parameter_gradient(block::DeviceKernelBlock, G)
    g = similar(G, Float64, block.nparams)
    launch_kernel!(gaussian_kernel_grad_kernel!, g, block.nparams, block.rows, block.X, block.D, G, block.γ)
end

"""
    get_device_hamiltonian(lmap, hr, phase, sp_diag)

Assemble the k-space Hamiltonians `Hk = Σ_R Hr[R] * phase[R, k]` from the real-space entries `hr` on the device.
"""
function get_device_hamiltonian(lmap::DeviceLinearMap, hr, phase, sp_diag)
    hk = similar(hr, ComplexF64, nnz_hk(lmap), size(phase, 2))
    launch_kernel!(assemble_hk_kernel!, hk, size(hk), lmap.grp_ptr, lmap.grp_ent, hr, lmap.Ridx, phase)
    return DeviceHk(lmap.Nε, lmap.hk_rowptr, lmap.hk_colval, lmap.hk_rowval, lmap.diag_pos, hk, sp_diag)
end

"""
    device_params(ham, ctx)

Device copies of the parameters of all models. They are only uploaded again if the parameters changed.
"""
function device_params(ham, ctx)
    return map(enumerate(ham.models)) do (m, model)
        params = get_params(model)
        cached = get(ctx.param_cache, m, nothing)
        if isnothing(cached) || cached[1] ≠ params
            cached = (copy(params), to_device(ctx.device, ComplexF64.(params)))
            ctx.param_cache[m] = cached
        end
        cached[2]
    end
end

device_phase(ctx, Rs, ks) = to_device(ctx.device, exp_2πi(Rs, ks))

"""
    device_phase(ctx, index, Rs, ks)

Device copy of the phase factors of structure `index`, cached for the k-points `ks` (fixed during training).
"""
function device_phase(ctx, index, Rs, ks)
    return get!(() -> device_phase(ctx, Rs, ks), ctx.phase_cache, (index, hash(ks)))
end

function get_device_hamiltonian(ham, index, ks)
    ctx = ham.device_ctx
    lmap = ctx.maps[index]
    hr = get_device_hr(lmap, device_params(ham, ctx))
    phase = device_phase(ctx, index, ham.Rs[index], ks)
    Hk = get_device_hamiltonian(lmap, hr, phase, ham.sp_diag)
    device_synchronize(Hk.nzval)
    return Hk, phase
end

"""
    densify(Hk::DeviceHk)

Convert a batch of sparse device Hamiltonians to a dense `(Nε, Nε, Nk)` device array.
"""
function densify(Hk::DeviceHk)
    H = fill!(similar(Hk.nzval, ComplexF64, Hk.Nε, Hk.Nε, nkpoints(Hk)), 0)
    launch_kernel!(densify_kernel!, H, size(Hk.nzval), Hk.rowval, Hk.colval, Hk.nzval)
end

"""
    host_hamiltonians(Hk)

Copy device Hamiltonians to the host as a vector of `SparseMatrixCSC` (sparse diagonalization) or `Matrix` (dense diagonalization).
"""
host_hamiltonians(Hk::AbstractVector{<:AbstractMatrix}) = Hk
function host_hamiltonians(Hk::DeviceHk)
    rowval = Array(Hk.rowval); colval = Array(Hk.colval); nzval = Array(Hk.nzval)
    Hks = [sparse(rowval, colval, nzval[:, k], Hk.Nε, Hk.Nε) for k in axes(nzval, 2)]
    return Hk.sp_diag isa Sparse ? Hks : Matrix.(Hks)
end

"""
    device_eigen(Hk::DeviceHk, sp_diag; Neig, target, method)

Diagonalize a batch of device Hamiltonians. Returns the eigenvalues as a host matrix `(Neig, Nk)` sorted in ascending order
and the eigenvectors as a device array `(Nε, Neig, Nk)`. This generic fallback diagonalizes on the host; device-specific
methods are provided by extensions.
"""
function device_eigen(Hk::DeviceHk, ::Dense; vectors=true, kwargs...)
    Es, vs = diagonalize(host_hamiltonians(Hk), vectors=vectors)
    return Es, copyto!(similar(Hk.nzval, ComplexF64, size(vs)), vs)
end

function device_eigen(Hk::DeviceHk, ::Sparse; Neig=6, target=0.0, method="shift-invert", vectors=true)
    Es, vs = diagonalize(host_hamiltonians(Hk), Neig=Neig, target=target, method=method)
    for k in axes(Es, 2)
        perm = sortperm(Es[:, k])
        Es[:, k] = Es[perm, k]; vs[:, :, k] = vs[:, perm, k]
    end
    return Es, copyto!(similar(Hk.nzval, ComplexF64, size(vs)), vs)
end

function diagonalize(Hk::DeviceHk; Neig=Hk.Nε, target=0, method="shift-invert", vectors=true, window=nothing)
    Es, vs = device_eigen(Hk, Hk.sp_diag, Neig=Neig, target=target, method=method, vectors=vectors)
    # Energy window of the dense diagonalization: the eigenvalues outside are removed afterwards
    isnothing(window) || Hk.sp_diag isa Sparse || return apply_window(Es, Array(vs), window)
    return Es, Array(vs)
end

"""
    DeviceParamGradient

Gradient of the loss of one structure w.r.t. the parameters of each model, computed on a device.
`grads[m]` is the (unregularized) gradient of model `m`.
"""
struct DeviceParamGradient
    grads :: Vector{Vector{Float64}}
end

"""
    get_device_param_gradient(ham, index, vs, dL_dE, phase)

Compute the gradient of the loss w.r.t. the model parameters from the eigenvectors `vs` and the gradient `dL_dE` w.r.t. the eigenvalues.
"""
function get_device_param_gradient(ham, index, vs, dL_dE, phase)
    ctx = ham.device_ctx
    lmap = ctx.maps[index]
    G = similar(vs, ComplexF64, nentries(lmap))
    w = to_device(ctx.device, Float64.(dL_dE))
    launch_kernel!(hellman_feynman_kernel!, G, length(G), lmap.I, lmap.J, lmap.Ridx, phase, vs, w)

    grads = [zeros(length(get_params(model))) for model in ham.models]
    for block in lmap.blocks
        grads[block.model_index] .+= Array(parameter_gradient(block, G))
    end
    return DeviceParamGradient(grads)
end

function device_forward(ham, index, loss, data::EigData)
    ctx = ham.device_ctx
    Hk, phase = get_device_hamiltonian(ham, index, data.kp)
    Neig = ham.sp_diag isa Sparse ? ctx.neig : Hk.Nε
    Es, vs = device_eigen(Hk, ham.sp_diag, Neig=Neig, target=ctx.target, method=ctx.method)
    L_train = loss(Es, data.Es)
    return L_train, (Es, vs, phase)
end

function device_backward(ham, index, loss, data::EigData, cache)
    Es, vs, phase = cache
    dL_dE = backward(loss, Es, data.Es)
    return get_device_param_gradient(ham, index, vs, dL_dE, phase)
end

"""
    RawModelGradient

Unregularized gradient of the loss w.r.t. the parameters of a single model, summed over a batch of structures.
"""
struct RawModelGradient
    grad :: Vector{Float64}
end

"""
    model_gradient_input(dL_dHr, model_index)

Select the input of `get_model_gradient` for the model `model_index`. Device gradients are already parameter gradients and are summed
over the batch; other gradients (w.r.t. the real-space Hamiltonian) are passed on unchanged.
"""
model_gradient_input(dL_dHr, model_index) = dL_dHr
function model_gradient_input(dL_dHr::AbstractVector{DeviceParamGradient}, model_index)
    return RawModelGradient(sum(g -> g.grads[model_index], dL_dHr))
end

function get_model_gradient(model::TBModel, indices, reg, g::RawModelGradient; soc=false)
    any(model.update) || return zeros(length(model.params))
    dV_penal = backward(reg, model.params)
    return @. ifelse(model.update, g.grad + dV_penal, 0.)
end

function get_model_gradient(kernel::HamiltonianKernel, indices, reg, g::RawModelGradient; soc=false)
    kernel.update || return zeros(length(kernel.params))
    return g.grad .+ backward(reg, kernel.params)
end

function get_model_gradient(kernel::HamiltonianKernelPrecalced, indices, reg, g::RawModelGradient; soc=false)
    kernel.update || return zeros(length(kernel.params))
    return kernel.kp.weights .* g.grad .+ backward(reg, kernel.params)
end

function get_model_gradient(soc_model::SOCModel, indices, reg, g::RawModelGradient; soc=true)
    dparams = soc_model.update ? g.grad : zeros(length(soc_model.params))
    return dparams + backward(reg, soc_model.params)
end

function get_model_gradient(ewald::EwaldOnsites, indices, reg, g::RawModelGradient; soc=true)
    dparams = ewald.update ? g.grad : zeros(length(ewald.params))
    return dparams + backward(reg, ewald.params)
end
