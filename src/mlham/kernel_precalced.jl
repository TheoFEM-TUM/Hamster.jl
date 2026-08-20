mutable struct HamiltonianKernelPrecalced{}
    params :: Vector{Float64}
    kp :: Kernelpoints
    sim_params :: Float64
    update :: Bool
    sm :: SimMat
end



function HamiltonianKernelPrecalced(kernel::HamiltonianKernel, rank, systems, conf = get_empty_config(); sim_tol = get_ml_sim_tol(conf), verbosity=get_verbosity(conf))
    if rank == 0 && verbosity > 1; println("    Rank 0 : Starting calculation of similarity matrix..."); end
    time = @elapsed begin
        kp, params = get_sorted_Kernelpoints(kernel.data_points, ones(Int64, length(kernel.data_points)), kernel.params, conf)
        sm = get_kernel_features(kernel.structure_descriptors, kp, kernel.sim_params, sim_tol, conf = conf, rank = rank, systems = systems)
    end
    if rank == 0 && verbosity > 1; println(@sprintf("    Finished calculating similarity matrix in %.2f seconds...", time)); end

    return HamiltonianKernelPrecalced(params, kp, kernel.sim_params, kernel.update, sm)
end



function get_hr(kernel::HamiltonianKernelPrecalced, mode, index; apply_soc=false)
    @views desc_vec = kernel.sm.feature_vec[index]
    (NR, Ne) = kernel.sm.feature_shape[1][index]
    key_ranges = kernel.kp.key_ranges
    Hr = get_empty_complex_hamiltonians(Ne, NR, mode)

    for R in 1:NR
        nnz_ham = desc_vec[R][2]

        # Per-task storage: each thread writes only to its own slot m, so no contention
        Is = Vector{Int}(undef, nnz_ham)
        Js = Vector{Int}(undef, nnz_ham)
        Vs = Vector{ComplexF64}(undef, nnz_ham)
        #keep = falses(nnz_ham)

        tforeach(1:nnz_ham) do m
            desc_vec_single, (i, j, key) = @views desc_vec[R][1][m]

            if !haskey(key_ranges, key)
                @warn "key not found in key_ranges, skipping" key=key R=R m=m
                return
            end

            params = @views kernel.params[key_ranges[key]]
            if length(desc_vec_single) != length(params)
                @warn "length mismatch, skipping" key=key R=R m=m len_desc=length(desc_vec_single) len_params=length(params)
                return
            end

            val = dot(desc_vec_single, params)

            Is[m] = i
            Js[m] = j
            Vs[m] = val
            #keep[m] = true

        end

        # Sequential, safe construction of the sparse matrix for this R
        #Hr[R] = sparse(Is[keep], Js[keep], Vs[keep], Ne, Ne)
        Hr[R] = sparse(Is, Js, Vs, Ne, Ne)
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
        # Write header to file
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_ml_sim_params(conf))
        println(file, "  env_scale = ", get_ml_env_scale(conf))
        println(file, "  apply_distortion = ", get_ml_apply_distortion(conf))
        println(file, "  apply_orthogonality = ", get_ml_apply_orthogonality(conf))
        println(file, "end")
        println(file, "")
        for n in eachindex(kernel.params)
            print(file, kernel.params[n])
            for data_point in data_points[n]
                print(file, " "); print(file, data_point)
            end
            print(file, "\n")
        end
    end
end

function write_params(params_data_points_tuple::Tuple{Vector{Float64}, Vector{SVector{8, Float64}}}, conf=get_empty_config(); filename=get_ml_filename(conf))
    open(filename*".dat", "w") do file
        # Write header to file
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_ml_sim_params(conf))
        println(file, "  env_scale = ", get_ml_env_scale(conf))
        println(file, "  apply_distortion = ", get_ml_apply_distortion(conf))
        println(file, "  apply_orthogonality = ", get_ml_apply_orthogonality(conf))
        println(file, "end")
        println(file, "")
        params, data_points = params_data_points_tuple
        for n in eachindex(params)
            print(file, params[n])
            for data_point in data_points[n]
                print(file, " "); print(file, data_point)
            end
            print(file, "\n")
        end
    end
end

function write_datapoints(data_points::Vector{SVector{8, Float64}}, target_dir::String, conf=get_empty_config(); filename=get_ml_filename(conf))
    open(joinpath(target_dir, filename*".dat"), "w") do file
        # Write header to file
        #params = init_ml_params!(data_points, conf)[1]
        Nparams = length(data_points)
        params = zeros(Nparams)
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_ml_sim_params(conf))
        println(file, "  env_scale = ", get_ml_env_scale(conf))
        println(file, "  apply_distortion = ", get_ml_apply_distortion(conf))
        println(file, "  apply_orthogonality = ", get_ml_apply_orthogonality(conf))
        println(file, "end")
        println(file, "")
        for n in eachindex(params)
            print(file, params[n])
            for data_point in data_points[n]
                print(file, " "); print(file, data_point)
            end
            print(file, "\n")
        end
    end
    println("Wrote datapoints to ", filename*".dat")
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
    key_ranges = kernel.kp.key_ranges
    weights = kernel.kp.weights
    if kernel.update
        for (bi, index) in enumerate(indices)
            
            for R in eachindex(dL_dHr[bi])
                @views desc_vec, nnz_ham = kernel.sm.feature_vec[index][R]
                tforeach(1:nnz_ham) do m
                    tid = Threads.threadid()
                    @views desc_vec_small, (i,j,key) =  desc_vec[m]

                    if !soc
                        dparams_threads[tid][key_ranges[key]]  .+= weights[key_ranges[key]] .* desc_vec_small .* real(dL_dHr[bi][R][i, j])
                    else
                        i1 = 2*i-1; j1 = 2*j-1
                        i2 = 2*i; j2 = 2*j
                        dparams_threads[tid][key_ranges[key]] .+= weights[key_ranges[key]] .* desc_vec_small .* real(dL_dHr[bi][R][i1, j1] + dL_dHr[bi][R][i2, j2])
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
