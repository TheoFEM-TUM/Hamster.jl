"""
    mutable struct HamiltonianKernel{T1, T2}

A kernel structure used for computing weighted similarity functions.

# Fields
- `ws :: Vector{Float64}`: Weights for each sample point.
- `xs :: Vector{T1}`: Sample points.
- `sim_params :: T2`: Parameters for the similarity function.
"""
mutable struct HamiltonianKernel{T1, T2, T3}
    params :: Vector{Float64}
    data_points :: Vector{T1}
    sim_params :: T2
    structure_descriptors :: Vector{T3}
    update :: Bool
end

"""
    HamiltonianKernel(strcs, bases, model, conf)

Constructor for a HamiltonianKernel model.
"""
function HamiltonianKernel(strcs::Vector{<:Structure}, bases::Vector{<:Basis}, model, comm, conf=get_empty_config(); verbosity=get_verbosity(conf),
    Ncluster=get_ml_ncluster(conf), Npoints=get_ml_npoints(conf), sim_params=get_sim_params(conf), update_ml=get_ml_update(conf), rank=0, nranks=1)
    
    structure_descriptors = map(eachindex(strcs)) do n
        get_tb_descriptor(model.hs[n], model.V, strcs[n], bases[n], conf)
    end
    if get_ml_init_params(conf)[1] ∈ ['r', 'z', 'o']
        Npoints_local = floor(Int64, Npoints / nranks)
        data_points_local = sample_structure_descriptors(reshape_structure_descriptors(structure_descriptors), Ncluster=Ncluster, Npoints=Npoints_local, ml_sampling=get_ml_sampling(conf))
        local_counts::Int32 = length(data_points_local)
        counts = MPI.Gather(local_counts, 0, comm)
        counts = MPI.bcast(counts, 0, comm)
        #data_points = MPI.Gatherv(data_points_local, counts, 0, comm)

        if rank == 0
            data_points_buf = MPI.VBuffer(similar(data_points_local, sum(counts)), counts)
        else
            data_points_buf = nothing
        end

        MPI.Gatherv!(view(data_points_local, 1:counts[rank + 1]), data_points_buf, 0, comm)
        data_points = rank == 0 ? data_points_buf.data : nothing
        data_points = MPI.bcast(data_points, comm)

        N_real = sum(counts)
        if N_real ≠ Npoints && rank == 0 && verbosity > 0
            @info "Number of samples changed from $Npoints to $N_real"
        end
    else
        _, data_points = read_ml_params(conf, filename=get_ml_init_params(conf))
    end
    params, data_points = init_ml_params!(data_points, conf)
    return HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update_ml)
end

exp_sim(x₁, x₂; σ=√0.05)::Float64 = exp(-normdiff(x₁, x₂)^2 / (2σ^2))

(k::HamiltonianKernel)(xin) = mapreduce(wx->wx[1]*exp_sim(wx[2], xin, σ=k.sim_params), +, zip(k.params, k.data_points))

"""
    get_hr(kernel::HamiltonianKernel, mode, index; apply_soc=false) -> Vector{Matrix{Float64}}

Constructs a set of real-space Hamiltonians from a `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel used for computing matrix elements.
- `mode`: Specifies the sparsity mode.
- `index`: Index specifying which structure to evaluate.

# Keyword Arguments
- `apply_soc`: If `true`, applies the spin-orbit coupling (SOC) basis transformation.

# Returns
- A vector of real-space Hamiltonian matrices, optionally modified with SOC transformations.
"""
function get_hr(kernel::HamiltonianKernel, mode::Dense, index; apply_soc=false)
    h_env = kernel.structure_descriptors[index]
    Hr = get_empty_complex_hamiltonians(size(h_env[1], 1), length(h_env), mode)
    for R in eachindex(h_env)
        for (i, j, hin) in zip(findnz(h_env[R])...)
            Hr[R][i, j] = kernel(hin)
        end
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

function get_hr(kernel::HamiltonianKernel, mode::Sparse, index; apply_soc=false)
    h_env = kernel.structure_descriptors[index]
    Nε = size(h_env[1], 1)
    Hr = get_empty_complex_hamiltonians(Nε, length(h_env), mode)
    for R in eachindex(h_env)
        is = Int64[]
        js = Int64[]
        vals = Float64[]
        for (i, j, hin) in zip(findnz(h_env[R])...)
            push!(is, i); push!(js, j); push!(vals, kernel(hin))
        end
        Hr[R] = sparse(is, js, vals, Nε, Nε)
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

"""
    update!(kernel::HamiltonianKernel, opt, grad)

Updates the parameters of a `HamiltonianKernel` using an optimization method `opt`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel whose parameters are to be updated.
- `opt`: The optimizer used to perform the update.
- `grad`: The gradient used for updating the parameters.
"""
function update!(kernel::HamiltonianKernel, opt, grad)
    update!(opt, kernel.params, grad)
end

"""
    get_params(kernel::HamiltonianKernel)

Retrieve the parameters associated with a `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel instance from which to extract parameters.

# Returns
- The parameters stored in the `ws` field of the given `HamiltonianKernel` instance.
"""
get_params(kernel::HamiltonianKernel) = kernel.params

"""
    write_params(kernel::HamiltonianKernel, conf=get_empty_config(); filename=get_ml_filename(conf))

Writes the parameters and configuration settings of a HamiltonianKernel object to a file.

# Arguments
- `kernel::HamiltonianKernel`: The HamiltonianKernel object containing the parameters and data points to write to the file.
- `conf`: A configuration object (default: `get_empty_config()`) containing simulation parameters and settings.
- `filename`: The name of the file to which the data will be written (default: `get_ml_filename(conf)`).
"""
function write_params(kernel::HamiltonianKernel, conf=get_empty_config(); filename=get_ml_filename(conf))
    open(filename*".dat", "w") do file
        # Write header to file
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_sim_params(conf))
        println(file, "  env_scale = ", get_env_scale(conf))
        println(file, "  apply_distortion = ", get_apply_distortion(conf))
        println(file, "end")
        println(file, "")
        for n in eachindex(kernel.params)
            print(file, kernel.params[n])
            for data_point in kernel.data_points[n]
                print(file, " "); print(file, data_point)
            end
            print(file, "\n")
        end
    end
end

"""
    read_ml_params(conf=get_empty_config(); filename=get_ml_filename(conf))

Reads the parameters for a HamiltonianKernel model from a file and returns the parameters and associated data points.

# Arguments
- `conf`: A configuration object (default: `get_empty_config()`) containing simulation parameters and settings.
- `filename`: The name of the `.dat` file to read from (default: `get_ml_filename(conf)`).
"""
function read_ml_params(conf=get_empty_config(); filename=get_ml_filename(conf))
    if !occursin(".dat", filename); filename *= ".dat"; end
    lines = open_and_read(filename)
    lines = split_lines(lines)
    N = length(lines[8]) - 1

    # Check that header params match Config
    @assert parse(Float64, lines[2][end]) == get_ml_rcut(conf)
    @assert parse(Float64, lines[3][end]) == get_sim_params(conf)
    @assert parse(Float64, lines[4][end]) == get_env_scale(conf)
    @assert parse(Bool, lines[5][end]) == get_apply_distortion(conf)

    data_points = SVector{N, Float64}[]
    params = Float64[]
    for line in lines[8:end]
        if length(line) > 1
            parsed_line = parse.(Float64, line)
            push!(params, parsed_line[1])
            push!(data_points, SVector{N, Float64}(parsed_line[2:end]))
        end
    end
    return params, data_points
end

"""
    init_ml_params!(data_points, conf=get_empty_config(); initas=get_ml_init_params(conf))

Initializes machine learning parameters based on a given initialization strategy and updates the `data_points`.

# Arguments
- `data_points`: The data points associated with the machine learning parameters.
- `conf`: A configuration object (default: `get_empty_config()`) containing simulation parameters and settings.
- `initas`: A string (default: `get_ml_init_params(conf)`) that specifies the initialization strategy. Possible values:
  - `'z'`: Initialize parameters to zeros.
  - `'o'`: Initialize parameters to ones.
  - `'r'`: Initialize parameters with random values.
  - `file`: Initialize parameters from a file `initas`
"""
function init_ml_params!(data_points, conf=get_empty_config(); initas=get_ml_init_params(conf))
    Nparams = length(data_points)
    if initas[1] == 'z'
        return zeros(Nparams), data_points
    elseif initas[1] == 'o'
        return ones(Nparams), data_points
    elseif initas[1] == 'r'
        return rand(Nparams), data_points
    else
        return read_ml_params(conf, filename=initas)
    end
end

"""
    set_params!(kernel::HamiltonianKernel, ws)

Set the parameters of a `HamiltonianKernel` instance.

# Arguments
- `kernel::HamiltonianKernel`: The kernel model whose parameters are to be updated.
- `ws`: The new parameter vector.

# Error Conditions
- Throws an error if the parameter vector `ws` is not of the correct size.

# Returns
- Updates the `Vs` field of the `kernel` in place if the consistency checks pass.
"""
function set_params!(kernel::HamiltonianKernel, params)
    throw_error = length(kernel.data_points) ≠ length(params)
    if throw_error
        error("Parameter vector is not of correct size ($(length(kernel.data_points)) ≠ $(length(params)))!")
    else
        kernel.params = params
    end
end

"""
    get_model_gradient(kernel::HamiltonianKernel, indices, reg, dL_dHr) -> Vector{Float64}

Computes the gradient of the model parameters for a given `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel for which the gradient is computed.
- `indices`: Indices specifying which structure descriptors to use.
- `reg`: Regularization term.
- `dL_dHr`: Gradient of the loss function with respect to the real-space Hamiltonian.

# Returns
- `dparams`: A vector containing the computed gradients of the model parameters.
"""
function get_model_gradient(kernel::HamiltonianKernel, indices, reg, dL_dHr)
    dparams = zeros(length(kernel.params))
    if kernel.update
        for n in eachindex(dparams)
            for index in indices
                h_env = kernel.structure_descriptors[index]
                for R in eachindex(dL_dHr[index]), (i, j, hin) in zip(findnz(h_env[R])...)
                    dparams[n] += exp_sim(kernel.data_points[n], hin, σ=kernel.sim_params) .* dL_dHr[index][R][i, j]
                end
            end
        end
        dparams_penal = backward(reg, kernel.params)
        return dparams .+ dparams_penal
    else 
        return dparams
    end
end