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
    HamiltonianKernel(strcs, bases, model, comm, conf; 
                      verbosity=get_verbosity(conf),
                      Ncluster=get_ml_ncluster(conf),
                      Npoints=get_ml_npoints(conf),
                      sim_params=get_ml_sim_params(conf),
                      update_ml=get_ml_update(conf),
                      mode=get_ml_mode(conf),
                      rank=0,
                      nranks=1)

Constructs a `HamiltonianKernel` model for machine learning-based Hamiltonian parameterization.

# Arguments
- `strcs`: Vector of `Structure` objects.
- `bases`: Vector of `Basis` objects corresponding to each structure.
- `model`: The underlying TB Hamiltonian model.
- `comm`: MPI communicator.
- `conf`: Configuration object.
- `verbosity`: Integer controlling logging output.
- `Ncluster`: Number of clusters for ML sampling.
- `Npoints`: Number of points to sample.
- `sim_params`: Kernel similarity measure.
- `update_ml`: Boolean flag controlling whether ML parameters are updated .
- `mode`: ML training mode, one of `"eval"`, `"refit"`, `"expand"`.
- `data_points`: Data points of another kernel (e.g. of the training set for the validation set). If given, no
  data points are sampled and the parameters are initialized as zeros (they are copied from that kernel).
- `rank`: MPI rank of the current process (default: 0).
- `nranks`: Total number of MPI ranks (default: 1).
- `label`, `write_summary`: Label of the model summary in `hamster.out` and whether it is written.

# Returns
A `HamiltonianKernel` object.

# Notes
- MPI is used for distributed sampling of data points; ensure that `comm`, `rank`, and `nranks` are consistent.
- The function supports evaluation-only mode (`mode="eval"`) where parameters are not updated.
- New or expanded parameters are initialized according to the `init_ml_params` configuration.
"""
function HamiltonianKernel(strcs::Vector{<:Structure}, bases::Vector{<:Basis}, model, comm, conf=get_empty_config(); 
                            verbosity=get_verbosity(conf),
                            Ncluster=get_ml_ncluster(conf),
                            Npoints=get_ml_npoints(conf),
                            sim_params=get_ml_sim_params(conf), 
                            update_ml=get_ml_update(conf),
                            mode=get_ml_mode(conf),
                            sim_mat = get_ml_sim_mat(conf),
                            data_points = nothing,
                            rank=0,
                            nranks=1,
                            label="",
                            write_summary=true)
    systems = [strc.system for strc in strcs]
    structure_descriptors = map(eachindex(strcs)) do n
        get_tb_descriptor(model.hs[n], model.params, strcs[n], bases[n], conf)
    end
    external_data_points = !isnothing(data_points)
    if get_load_model(conf)
        params, data_points = model_ml_parameters(conf)
    elseif external_data_points
        params = zeros(length(data_points))
    elseif get_init_mode(get_ml_init_params(conf)) ≠ :file
        Npoints_local = floor(Int64, Npoints / nranks)
        data_points_local = sample_structure_descriptors(reshape_structure_descriptors(structure_descriptors), Ncluster=Ncluster, Npoints=Npoints_local, ml_sampling=get_ml_sampling(conf))
        local_counts::Int32 = length(data_points_local)
        counts = MPI.Gather(local_counts, 0, comm)
        counts = MPI.bcast(counts, 0, comm)

        data_points_buf = nothing
        if rank == 0
            data_points_buf = MPI.VBuffer(similar(data_points_local, sum(counts)), counts)
        end

        MPI.Gatherv!(view(data_points_local, 1:counts[rank + 1]), data_points_buf, 0, comm)
        data_points = rank == 0 ? data_points_buf.data : nothing
        data_points = MPI.bcast(data_points, comm)

        N_real = sum(counts)
        # COV_EXCL_START
        if N_real ≠ Npoints && rank == 0 && verbosity > 0
            @info "Number of samples changed from $Npoints to $N_real"
        end
        # COV_EXCL_STOP
    end
    if !external_data_points && !get_load_model(conf)
        params, data_points = init_ml_params!(data_points, conf, mode=mode, update_ml=update_ml)
    end
    kernel = HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update_ml)
    write_summary && write_model_summary(kernel, strcs, bases, comm, conf, rank=rank, label=label)
    return sim_mat ? HamiltonianKernelPrecalced(kernel, rank, systems, conf) : kernel
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
        is, js, hins = findnz(h_env[R])
        vals = evaluate_kernel(kernel, hins)
        for t in eachindex(vals)
            Hr[R][is[t], js[t]] = vals[t]
        end
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

function get_hr(kernel::HamiltonianKernel, mode::Sparse, index; apply_soc=false)
    h_env = kernel.structure_descriptors[index]
    Nε = size(h_env[1], 1)
    Hr = get_empty_complex_hamiltonians(Nε, length(h_env), mode)
    for R in eachindex(h_env)
        is, js, hins = findnz(h_env[R])
        Hr[R] = sparse(is, js, evaluate_kernel(kernel, hins), Nε, Nε)
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

"""
    evaluate_kernel(kernel::HamiltonianKernel, hins) -> Vector{Float64}

Evaluate the kernel model for every descriptor in `hins` (in parallel).
"""
function evaluate_kernel(kernel::HamiltonianKernel, hins)
    vals = Vector{Float64}(undef, length(hins))
    tforeach(eachindex(hins)) do t
        vals[t] = kernel(hins[t])
    end
    return vals
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
    copy_params!(receiving_model::HamiltonianKernel, sending_model::HamiltonianKernel) -> Nothing

Copy parameters from one Hamiltonian kernel to another.

# Arguments
- `receiving_model::HamiltonianKernel`: The kernel that will receive the parameters.
- `sending_model::HamiltonianKernel`: The kernel providing the parameters.
"""
function copy_params!(receiving_model::HamiltonianKernel, sending_model::HamiltonianKernel)
    set_params!(receiving_model, get_params(sending_model))
end

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
        for key in ML_HEADER_TAGS
            value = ml_setting(conf, key)
            value isa AbstractVector && isempty(value) && continue
            println(file, "  $key = ", value isa AbstractVector ? join(value, " ") : value)
        end
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
    ML_HEADER_TAGS

The tags of the ML block that define an ML model, written to the header of its parameter file (`write_params`).
"""
const ML_HEADER_TAGS = ["rcut", "sim_params", "env_scale", "strc_scale", "apply_distortion", "apply_distance_distortion",
                        "apply_orthogonality", "key_dims"]

ml_setting(conf, key) = getfield(@__MODULE__, Symbol("get_ml_", key))(conf)

"""
    read_ml_header(filename) -> Dict{String, String}

The settings in the header (`begin` ... `end`) of the ML parameter file `filename` (with or without `.dat`).
"""
function read_ml_header(filename)
    endswith(filename, ".dat") || (filename *= ".dat")
    header = Dict{String, String}()
    for line in eachline(filename)
        words = split(line)
        isempty(words) && continue
        words[1] == "begin" && continue
        words[1] == "end" && break
        occursin('=', line) || continue
        key, value = strip.(split(line, '=', limit=2))
        header[lowercase(key)] = value
    end
    return header
end

"""
    same_ml_setting(value, text)

Whether the setting `value` (from the config) equals `text` (from the header of a parameter file); numbers are compared
with a relative tolerance of 1e-10.
"""
function same_ml_setting(value, text)
    other = convert_value(text)
    value isa AbstractVector && !(other isa AbstractVector) && (other = [other])
    value isa AbstractVector && return length(value) == length(other) && all(same_ml_setting.(value, string.(other)))
    value isa Bool && return other isa Bool && value == other
    value isa Number && other isa Number && return isapprox(value, other, rtol=1e-10)
    return string(value) == string(other)
end

"""
    check_ml_settings(conf, filename)

Throw an error if a setting in the header of the ML parameter file `filename` (see `ML_HEADER_TAGS`) differs from the
value in `conf`.
"""
function check_ml_settings(conf, filename)
    for (key, text) in read_ml_header(filename)
        key ∈ ML_HEADER_TAGS || continue
        value = ml_setting(conf, key)
        same_ml_setting(value, text) ||
            error("The ML model in $filename was fitted with $key = $text, but $key = $(value isa AbstractVector ? join(value, " ") : value) " *
                  "in the ML block. Remove $key from the ML block to use the value of the file.")
    end
    return nothing
end

"""
    apply_ml_settings!(conf)

Use a fitted ML model with the settings it was fitted with: the settings in the header of the ML parameter files
(`init_params` of the ML block if it is a file, and `filename` with `mode = expand`) that are not set in the ML block
are taken from the file; settings that are set and differ raise an error (`check_ml_settings`). Files written before
the header contained all settings of `ML_HEADER_TAGS` keep the defaults for the missing ones.
"""
function apply_ml_settings!(conf)
    get_load_model(conf) && return nothing
    files = String[]
    get_init_mode(get_ml_init_params(conf)) == :file && push!(files, get_ml_init_params(conf))
    get_ml_mode(conf) == "expand" && push!(files, get_ml_filename(conf))
    for file in files
        for (key, text) in read_ml_header(file)
            key ∈ ML_HEADER_TAGS || continue
            conf(key, "ML") == "default" && set_value!(conf, key, "ML", text)
        end
        check_ml_settings(conf, endswith(file, ".dat") ? file : file * ".dat")
    end
    return nothing
end

"""
    read_ml_params(conf=get_empty_config(); filename=get_ml_filename(conf))

Reads the parameters for a HamiltonianKernel model from a file and returns the parameters and associated data points.

# Arguments
- `conf`: A configuration object (default: `get_empty_config()`) containing simulation parameters and settings.
- `filename`: The name of the `.dat` file to read from (default: `get_ml_filename(conf)`).
"""
function read_ml_params(conf=get_empty_config(); filename=get_ml_filename(conf))
    endswith(filename, ".dat") || (filename *= ".dat")
    lines = open_and_read(filename)
    lines = split_lines(lines)

    check_ml_settings(conf, filename)
    data_start = something(findfirst(line -> !isempty(line) && line[1] == "end", lines), -1) + 2

    N = length(lines[data_start]) - 1
    data_points = SVector{N, Float64}[]
    params = Float64[]
    for line in lines[data_start:end]
        if length(line) > 1
            parsed_line = parse.(Float64, line)
            push!(params, parsed_line[1])
            push!(data_points, SVector{N, Float64}(parsed_line[2:end]))
        end
    end
    return params, data_points
end

"""
    init_ml_params!(data_points, conf=get_empty_config(); 
                   initas=get_ml_init_params(conf),
                   filename=get_ml_filename(conf),
                   mode=get_ml_mode(conf),
                   update_ml=get_ml_update(conf))

Initializes ML parameters.

# Arguments
- `data_points`: An array of data points associated with the ML parameters.
- `conf`: A configuration object.
- `initas`: A string specifying the initialization strategy for **new parameters** (see [`get_init_mode`](@ref)):
  - `"zeros"` or `"z"` : Initialize parameters to zeros.
  - `"ones"` or `"o"`  : Initialize parameters to ones.
  - `"rand"` or `"r"`  : Initialize parameters with random values.
  - any other value     : Load parameters from the file `initas`.
- `filename`: A string specifying the file from which old parameters can be loaded when `mode="expand"`.
- `mode`: A string (default: `get_ml_mode(conf)`) specifying the ML training mode:
  - `"refit"`  : Train from scratch.
  - `"eval"`   : Evaluation mode.
  - `"expand"` : Load old parameters from `filename` and append new parameters initialized according to `initas`.
- `update_ml`: Boolean controlling whether ML parameters are updated during training.

# Returns
- `params`: Array of initialized ML parameters.
- `data_points`: (Updated) array of data points.
"""
function init_ml_params!(data_points, conf=get_empty_config(); 
                            initas=get_ml_init_params(conf),
                            filename=get_ml_filename(conf),
                            mode=get_ml_mode(conf),
                            update_ml=get_ml_update(conf))

    Nparams = isnothing(data_points) ? 1 : length(data_points)

    init_mode = get_init_mode(initas)
    if init_mode == :zeros
        params = zeros(Nparams)
    elseif init_mode == :ones
        params = ones(Nparams)
    elseif init_mode == :rand
        params = rand(Nparams)
    else
        params, data_points = read_ml_params(conf, filename=initas)
    end
    if mode == "expand"
        params_old, data_points_old = read_ml_params(conf, filename=filename)
        params = vcat(params_old, params)
        data_points = vcat(data_points_old, data_points)
    end
    return params, data_points
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
function get_model_gradient(kernel::HamiltonianKernel, indices, reg, dL_dHr; soc=false)
    dparams = zeros(length(kernel.params))
    if kernel.update
        # Collect the descriptors and the gradient w.r.t. their matrix elements once (instead of once per parameter)
        hins = eltype(eltype(kernel.structure_descriptors[first(indices)]))[]
        gs = Float64[]
        for (bi, index) in enumerate(indices)
            h_env = kernel.structure_descriptors[index]
            for R in eachindex(dL_dHr[bi])
                for (i, j, hin) in zip(findnz(h_env[R])...)
                    push!(hins, hin)
                    if !soc
                        push!(gs, real(dL_dHr[bi][R][i, j]))
                    else
                        push!(gs, real(dL_dHr[bi][R][2i-1, 2j-1] + dL_dHr[bi][R][2i, 2j]))
                    end
                end
            end
        end
        tforeach(eachindex(dparams)) do n
            acc = 0.0
            for t in eachindex(hins)
                acc += exp_sim(kernel.data_points[n], hins[t], σ=kernel.sim_params) * gs[t]
            end
            dparams[n] = acc
        end
        dparams_penal = backward(reg, kernel.params)
        return dparams .+ dparams_penal
    else 
        return dparams
    end
end
