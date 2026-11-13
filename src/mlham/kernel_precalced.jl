"""
    mutable struct HamiltonianKernelPrecalced

A kernel model with precalculated similarity matrices (`sim_mat = true`, default), i.e., the similarities of the descriptors of all
matrix elements to the data points are computed once and stored as `SimMat`s (see `HamiltonianKernel` for the model that
evaluates them on the fly).

# Fields
- `params :: Vector{Float64}`: The ML model parameters.
- `kp :: Kernelpoints`: Kernel points data including sampling points and key ranges.
- `sim_params :: Float64`: Parameters for the similarity function.
- `update :: Bool`: Flag indicating whether parameters should be updated during optimization.
- `sms :: Vector{SimMat}`: Precomputed kernel features, one `SimMat` per structure.
- `simmat_file :: String`: HDF5 file the `SimMat`s are written to / loaded from when they are batched (`nbatch > 1`).
"""
mutable struct HamiltonianKernelPrecalced
    params :: Vector{Float64}
    kp :: Kernelpoints
    sim_params :: Float64
    update :: Bool
    sms :: Vector{SimMat}
    simmat_file :: String
end

HamiltonianKernelPrecalced(params, kp, sim_params, update, sms) = HamiltonianKernelPrecalced(params, kp, sim_params, update, sms, simmat_filename(0))

"""
    HamiltonianKernelPrecalced(kernel::HamiltonianKernel, rank, systems, conf=get_empty_config();
                               weights=ones(Int64, length(kernel.data_points)), tol=get_sp_tol(conf), simmat_prefix="simmat", verbosity=get_verbosity(conf))

Constructs a `HamiltonianKernelPrecalced` by sorting the data points of `kernel` by key (`Kernelpoints`, with the multiplicities
`weights` of the data points) and precomputing the similarity matrices of its descriptors. The `SimMat`s are labeled by
`simmat_prefix` (and written to file with `nbatch > 1`).
"""
function HamiltonianKernelPrecalced(kernel::HamiltonianKernel, rank, systems, conf=get_empty_config();
                                    weights=ones(Int64, length(kernel.data_points)), tol=get_sp_tol(conf), simmat_prefix="simmat",
                                    verbosity=get_verbosity(conf))
    if rank == 0 && verbosity > 1; println("    Rank 0 : Starting calculation of similarity matrix..."); end
    time = @elapsed begin
        kp, params = get_sorted_Kernelpoints(kernel.data_points, weights, kernel.params, conf)
        if get_verbosity(conf) >= 2
            #ok = check_consistency(kp.datapoints, params; key_ranges=kp.key_ranges)
            #ok2 = verify_kernelpoints(kp, conf)
        end
        precalced = HamiltonianKernelPrecalced(params, kp, kernel.sim_params, kernel.structure_descriptors, kernel.update, tol;
                                               conf=conf, rank=rank, systems=systems, simmat_prefix=simmat_prefix)
    end
    if rank == 0 && verbosity > 1; println(@sprintf("    Finished calculating similarity matrix in %.2f seconds...", time)); end
    return precalced
end

"""
    HamiltonianKernelPrecalced(params, kp, sim_params, structure_descriptors, update, tol; conf, rank, systems, simmat_prefix) -> HamiltonianKernelPrecalced
"""
function HamiltonianKernelPrecalced(params :: Vector{Float64},
    kp,
    sim_params,
    structure_descriptors,
    update :: Bool,
    tol :: Float64;
    conf = get_empty_config(),
    rank = 0,
    systems = nothing,
    simmat_prefix = "simmat"
    )

    nbatch = get_nbatch(conf)
    label = replace(simmat_prefix, r"^simmat_?" => "")
    if nbatch > 1 && !isempty(structure_descriptors)
        # compute, write and drop the kernel features batch by batch to limit the peak memory
        simmat_file = simmat_filename(rank; prefix = simmat_prefix)
        sms = nothing
        for (batch_id, inds) in enumerate(chunks(1:length(structure_descriptors), n = nbatch))
            sms_batch = get_kernel_features(structure_descriptors[inds], kp, sim_params, tol, conf = conf, rank = rank,
                                            systems = systems === nothing ? nothing : systems[inds], label = label)
            write_simmat(sms_batch; filename = simmat_file, append = batch_id > 1)
            sms_empty = [get_empty_simmat(s) for s in sms_batch]
            sms = sms === nothing ? sms_empty : append!(sms, sms_empty)
            sms_batch = nothing
            GC.gc()
        end
    else
        sms = get_kernel_features(structure_descriptors, kp, sim_params, tol, conf = conf, rank = rank, systems = systems, label = label)
        simmat_file = simmat_filename(rank; prefix = simmat_prefix)
    end
    return HamiltonianKernelPrecalced(params,kp, sim_params, update, sms, simmat_file)
end

"""
    unload_simmats!(kernel::HamiltonianKernelPrecalced, indices)

Replaces `kernel.sms[i]` for all `i` in `indices` by empty `SimMat`s (see `get_empty_simmat`) to free memory.
"""
function unload_simmats!(kernel::HamiltonianKernelPrecalced, indices)
    for i in indices
        kernel.sms[i] = get_empty_simmat(kernel.sms[i])
    end
    return kernel
end

"""
    load_simmats!(kernel::HamiltonianKernelPrecalced, indices; filename=kernel.simmat_file)

Replaces `kernel.sms[i]` for all `i` in `indices` by the `SimMat`s read with `read_simmat_index` from `filename`.
"""
function load_simmats!(kernel::HamiltonianKernelPrecalced, indices; filename = kernel.simmat_file)
    for i in indices
        kernel.sms[i] = read_simmat_index(i; filename = filename)
    end
    return kernel
end

"""
    get_hr(kernel::HamiltonianKernelPrecalced, mode, index; apply_soc=false) -> Vector{Matrix{Float64}}

Constructs a set of real-space Hamiltonians from a `HamiltonianKernelPrecalced`.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The Hamiltonian kernel used for computing matrix elements.
- `mode`: Specifies the sparsity mode.
- `index`: Index specifying which structure to evaluate.

# Keyword Arguments
- `apply_soc`: If `true`, applies the spin-orbit coupling (SOC) basis transformation.

# Returns
- A vector of real-space Hamiltonian matrices, optionally modified with SOC transformations.
"""
function get_hr(kernel::HamiltonianKernelPrecalced, mode, index; apply_soc=false)
    sm = kernel.sms[index]
    (NR, Ne) = sm.feature_shape[1]
    params = kernel.params
    Hr = get_empty_complex_hamiltonians(Ne, NR, mode)
    for R in 1:NR
        block = sm.blocks[R]
        Vs = Vector{ComplexF64}(undef, nentries(block))
        tforeach(eachindex(Vs)) do m
            s = 0.0
            @inbounds for q in block.ptr[m]:block.ptr[m+1]-1
                s += block.val[q] * params[block.idx[q]]
            end
            Vs[m] = s
        end
        Hr[R] = sparse(Vector{Int}(block.is), Vector{Int}(block.js), Vs, Ne, Ne)
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

"""
    update!(kernel::HamiltonianKernelPrecalced, opt, grad)

Updates the parameters of a `HamiltonianKernelPrecalced` using an optimization method `opt`.
"""
function update!(kernel::HamiltonianKernelPrecalced, opt, grad)
    update!(opt, kernel.params, grad)
end

"""
    get_params(kernel::HamiltonianKernelPrecalced)

Retrieve the parameters associated with a `HamiltonianKernelPrecalced`.
"""
get_params(kernel::HamiltonianKernelPrecalced) = kernel.params

"""
    copy_params!(receiving_model::HamiltonianKernelPrecalced, sending_model::HamiltonianKernelPrecalced)

Copy parameters from one `HamiltonianKernelPrecalced` model to another.
"""
function copy_params!(receiving_model::HamiltonianKernelPrecalced, sending_model::HamiltonianKernelPrecalced)
    set_params!(receiving_model, get_params(sending_model))
end

"""
    write_params(kernel::HamiltonianKernelPrecalced, conf=get_empty_config(); filename=get_ml_filename(conf))

Writes the parameters and data points of a `HamiltonianKernelPrecalced` to a file (same format as for a `HamiltonianKernel`).
"""
function write_params(kernel::HamiltonianKernelPrecalced, conf=get_empty_config(); filename=get_ml_filename(conf))
    write_params((kernel.params, kernel.kp.datapoints), conf, filename=filename)
end

"""
    set_params!(kernel::HamiltonianKernelPrecalced, params)

Set the parameters of a `HamiltonianKernelPrecalced` instance.
"""
function set_params!(kernel::HamiltonianKernelPrecalced, params)
    kernel.params = params
end

"""
    get_model_gradient(kernel::HamiltonianKernelPrecalced, indices, reg, dL_dHr) -> Vector{Float64}

Computes the gradient of the model parameters for a given `HamiltonianKernelPrecalced`.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The Hamiltonian kernel for which the gradient is computed.
- `indices`: Indices specifying which structure descriptors to use.
- `reg`: Regularization term.
- `dL_dHr`: Gradient of the loss function with respect to the real-space Hamiltonian.

# Returns
- `dparams`: A vector containing the computed gradients of the model parameters.
"""
function get_model_gradient(kernel::HamiltonianKernelPrecalced, indices, reg, dL_dHr; soc=false)
    Nparams = length(kernel.params)
    kernel.update || return zeros(Nparams)
    # The (structure, R) blocks of the batch; the entries of all blocks are numbered consecutively
    blocks = [(bi, R) for bi in eachindex(indices) for R in eachindex(dL_dHr[bi])]
    block_ends = cumsum([nentries(kernel.sms[indices[bi]].blocks[R]) for (bi, R) in blocks])
    Nentries = isempty(block_ends) ? 0 : last(block_ends)

    # Every task accumulates the (unweighted) contributions of a contiguous range of entries into its own buffer
    ntasks = clamp(cld(Nentries, 4096), 1, Threads.nthreads())
    buffers = tmap(chunks(1:Nentries, n=ntasks)) do entry_range
        dparams_task = zeros(Nparams)
        isempty(entry_range) && return dparams_task
        for b in searchsortedfirst(block_ends, first(entry_range)):searchsortedfirst(block_ends, last(entry_range))
            bi, R = blocks[b]
            block_start = b == 1 ? 1 : block_ends[b-1] + 1
            ms = (max(first(entry_range), block_start):min(last(entry_range), block_ends[b])) .- (block_start - 1)
            accumulate_kernel_gradient!(dparams_task, kernel.sms[indices[bi]].blocks[R], ms, dL_dHr[bi][R], soc)
        end
        return dparams_task
    end

    # Sum the buffers (in a fixed order), apply the weights of the kernel points and add the regularization
    dparams = backward(reg, kernel.params)
    weights = kernel.kp.weights
    tforeach(chunks(1:Nparams, n=Threads.nthreads())) do param_range
        @inbounds for n in param_range
            s = 0.0
            for dparams_task in buffers
                s += dparams_task[n]
            end
            dparams[n] += weights[n] * s
        end
    end
    return dparams
end

"""
    accumulate_kernel_gradient!(dparams, block::SimMatBlock, ms, dL_dHr_R, soc)

Add the gradient contributions `feature * real(dL_dHr_R[i, j])` of the entries `ms` of one `R` block to `dparams`.
"""
function accumulate_kernel_gradient!(dparams, block::SimMatBlock, ms, dL_dHr_R, soc)
    @inbounds for m in ms
        i, j = block.is[m], block.js[m]
        g = soc ? real(dL_dHr_R[2i-1, 2j-1] + dL_dHr_R[2i, 2j]) : real(dL_dHr_R[i, j])
        for q in block.ptr[m]:block.ptr[m+1]-1
            dparams[block.idx[q]] += block.val[q] * g
        end
    end
    return dparams
end
