"""
    mutable struct HamiltonianKernelPrecalced

A precalculated kernel structure for efficient Hamiltonian parameterization with precomputed similarity matrices.

# Fields
- `params :: Vector{Float64}`: The ML model parameters.
- `kp :: Kernelpoints`: Kernel points data including sampling points and key ranges.
- `sim_params :: Float64`: Parameters for the similarity function.
- `update :: Bool`: Flag indicating whether parameters should be updated during optimization.
- `sm :: SimMat`: Precomputed similarity matrix containing feature vectors and shapes.
"""
mutable struct HamiltonianKernelPrecalced{T1, T2}
    params :: Vector{Float64}
    kp :: T1
    sim_params :: Float64
    update :: Bool
    sm :: T2
end



"""
    HamiltonianKernelPrecalced(kernel::HamiltonianKernel, rank, systems, conf; 
                               sim_tol=get_ml_sim_tol(conf), 
                               verbosity=get_verbosity(conf))

Constructs a `HamiltonianKernelPrecalced` object by precomputing the similarity matrix from a `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernel`: The base kernel from which to construct the precalculated kernel.
- `rank`: MPI rank of the current process.
- `systems`: Vector of system objects for distributed computation.
- `conf`: Configuration object (default: `get_empty_config()`).
- `sim_tol`: Similarity matrix tolerance (default: `get_ml_sim_tol(conf)`).
- `verbosity`: Logging verbosity level (default: `get_verbosity(conf)`).

# Returns
A `HamiltonianKernelPrecalced` object with precomputed similarity matrices for efficient Hamiltonian evaluations.

# Notes
- The constructor automatically computes sorted kernel points and feature vectors.
- MPI synchronization is used for distributed similarity matrix computation.
- Verbosity > 1 will print timing information for the similarity matrix calculation.
"""
function HamiltonianKernelPrecalced(kernel::HamiltonianKernel, rank, systems, conf = get_empty_config(); sim_tol = get_ml_sim_tol(conf), verbosity=get_verbosity(conf))
    if rank == 0 && verbosity > 1; println("    Rank 0 : Starting calculation of similarity matrix..."); end
    time = @elapsed begin
        kp, params = get_sorted_Kernelpoints(kernel.data_points, ones(Int64, length(kernel.data_points)), kernel.params, conf)
        sm = get_kernel_features(kernel.structure_descriptors, kp, kernel.sim_params, sim_tol, conf = conf, rank = rank, systems = systems)
    end
    if rank == 0 && verbosity > 1; println(@sprintf("    Finished calculating similarity matrix in %.2f seconds...", time)); end

    return HamiltonianKernelPrecalced(params, kp, kernel.sim_params, kernel.update, sm)
end



"""
    get_hr(kernel::HamiltonianKernelPrecalced, mode, index; apply_soc=false)

Retrieve the real-space Hamiltonian matrix for a given structure using precomputed kernel features.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The precalculated kernel containing precomputed feature vectors.
- `mode`: The Hamiltonian mode (e.g., number of orbitals or electronic basis).
- `index`: Index of the structure for which to retrieve the Hamiltonian.
- `apply_soc`: Boolean flag to apply spin-orbit coupling transformation (default: `false`).

# Returns
- `Hr`: Vector of sparse complex Hamiltonian matrices, one for each real-space lattice vector.

# Notes
- The Hamiltonian is constructed by computing dot products between descriptor vectors and learned parameters.
- Per-thread storage is used to avoid contention in the parallel loop over non-zero elements.
- If `apply_soc=true`, the spin-orbit coupling basis transformation is applied to each matrix.
"""
function get_hr(kernel::HamiltonianKernelPrecalced, mode, index; apply_soc=false)
    blocks = kernel.sm.feature_vec[index]
    (NR, Ne) = kernel.sm.feature_shape[1][index]
    params = kernel.params
    Hr = get_empty_complex_hamiltonians(Ne, NR, mode)
    for R in 1:NR
        b = blocks[R]
        Vs = Vector{ComplexF64}(undef, nelements(b))
        tforeach(1:nelements(b)) do m
            s = 0.0
            @inbounds for k in b.ptr[m]:b.ptr[m+1]-1
                s += b.val[k] * params[b.idx[k]]
            end
            Vs[m] = s
        end
        Hr[R] = sparse(Int.(b.i), Int.(b.j), Vs, Ne, Ne)
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end



"""
    update!(kernel::HamiltonianKernelPrecalced, opt, grad)

Updates the parameters of a `HamiltonianKernel` using an optimization method `opt`.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The Hamiltonian kernel whose parameters are to be updated.
- `opt`: The optimizer used to perform the update.
- `grad`: The gradient used for updating the parameters.
"""
function update!(kernel::HamiltonianKernelPrecalced, opt, grad)
    update!(opt, kernel.params, grad)
end

"""
    get_params(kernel::HamiltonianKernelPrecalced)

Retrieve the parameters associated with a `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The Hamiltonian kernel instance from which to extract parameters.

# Returns
- The parameters stored in the `ws` field of the given `HamiltonianKernel` instance.
"""
get_params(kernel::HamiltonianKernelPrecalced) = kernel.params

"""
    copy_params!(receiving_model::HamiltonianKernelPrecalced, sending_model::HamiltonianKernelPrecalced)

Copy parameters from one `HamiltonianKernelPrecalced` model to another.

# Arguments
- `receiving_model::HamiltonianKernelPrecalced`: The destination kernel model to receive parameters.
- `sending_model::HamiltonianKernelPrecalced`: The source kernel model to copy parameters from.

# Notes
- This function modifies `receiving_model` in place by updating its `params` field.
"""
function copy_params!(receiving_model::HamiltonianKernelPrecalced, sending_model::HamiltonianKernelPrecalced)
    set_params!(receiving_model, get_params(sending_model))
end


"""
    write_params(kernel::HamiltonianKernelPrecalced, conf=get_empty_config(); filename=get_ml_filename(conf))

Writes the parameters and configuration settings of a HamiltonianKernel object to a file.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The HamiltonianKernel object containing the parameters and data points to write to the file.
- `conf`: A configuration object (default: `get_empty_config()`) containing simulation parameters and settings.
- `filename`: The name of the file to which the data will be written (default: `get_ml_filename(conf)`).
"""
function write_params(kernel::HamiltonianKernelPrecalced, conf=get_empty_config(); filename=get_ml_filename(conf))
    data_points = kernel.kp.datapoints
    open(filename*".dat", "w") do file
        write_ml_header(file, conf)
        for n in eachindex(kernel.params)
            print(file, kernel.params[n])
            for data_point in data_points[n]
                print(file, " "); print(file, data_point)
            end
            print(file, "\n")
        end
    end
end



"""
    set_params!(kernel::HamiltonianKernelPrecalced, ws)

Set the parameters of a `HamiltonianKernel` instance.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The kernel model whose parameters are to be updated.
- `ws`: The new parameter vector.

# Error Conditions
- Throws an error if the parameter vector `ws` is not of the correct size.

# Returns
- Updates the `Vs` field of the `kernel` in place if the consistency checks pass.
"""
function set_params!(kernel::HamiltonianKernelPrecalced, params)
    #throw_error = length(kernel.data_points) ≠ length(params)
    #if throw_error
    #    error("Parameter vector is not of correct size ($(length(kernel.data_points)) ≠ $(length(params)))!")
    #else
        kernel.params = params
    #end
end

"""
    get_model_gradient(kernel::HamiltonianKernelPrecalced, indices, reg, dL_dHr) -> Vector{Float64}

Computes the gradient of the model parameters for a given `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernelPrecalced`: The Hamiltonian kernel for which the gradient is computed.
- `indices`: Indices specifying which structure descriptors to use.
- `reg`: Regularization term.
- `dL_dHr`: Gradient of the loss function with respect to the real-space Hamiltonian.

# Returns
- `dparams`: A vector containing the computed gradients of the model parameters.
"""
function get_model_gradient(kernel::HamiltonianKernelPrecalced, indices, reg, dL_dHr; soc=false)
    dparams = zeros(length(kernel.params))
    nt = Threads.maxthreadid() 
    dparams_threads = [zeros(length(kernel.params)) for _ in 1:nt]
    weights = kernel.kp.weights
    if kernel.update
        for (bi, index) in enumerate(indices)
            for R in eachindex(dL_dHr[bi])
                b = kernel.sm.feature_vec[index][R]
                tforeach(1:nelements(b)) do m
                    dp = dparams_threads[Threads.threadid()]
                    i, j = b.i[m], b.j[m]
                    g = soc ? real(dL_dHr[bi][R][2i-1, 2j-1] + dL_dHr[bi][R][2i, 2j]) : real(dL_dHr[bi][R][i, j])
                    @inbounds for k in b.ptr[m]:b.ptr[m+1]-1
                        n = b.idx[k]
                        dp[n] += weights[n] * b.val[k] * g
                    end
                end
            end
        end
        for dp in dparams_threads
            dparams .+= dp
        end
        dparams_penal = backward(reg, kernel.params)
        return dparams .+ dparams_penal
    else 
        return dparams
    end
end
