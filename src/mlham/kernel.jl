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
    sim_params :: Float64
    update :: Bool
    feature_vec :: Vector{T3}
    feature_shape :: Tuple{Vector{T2}, Int64}
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

function get_kernel_features(structure_descriptors, data_points, sim_params, tol = 1e-8; conf = get_empty_config(), rank = 0)
    verbosity = get_verbosity(conf)
    #println(tol)
    #println("NTHREADS",Threads.nthreads())
    N_mats = size(structure_descriptors)[1]
    N_dp = size(data_points)[1]
    descr_sizes = [(size(structure_descriptors[i])[1], size(structure_descriptors[i][1])[1]) for i in 1:N_mats]
    Desc_Vec = [ [[ spzeros(Float64, descr_sizes[i][2], descr_sizes[i][2]) for _ in 1:descr_sizes[i][1] ] for d in 1:N_dp]
      for i in 1:N_mats ]
    #N_test = 0
    tforeach(1:N_mats) do i
        @views h_env = structure_descriptors[i]
        for d in 1:N_dp
            @views data_point = data_points[d]
            N_R, Ne = descr_sizes[i]
            for R in 1:N_R
                is = Vector{Int32}() 
                js = Vector{Int32}() 
                vals = Vector{Float64}() 
                for (i_mat, j_mat, hin) in zip(findnz(h_env[R])...)
                    val = exp_sim(data_point, hin, σ=sim_params)
                    if abs(val) > tol
                        push!(is, i_mat)
                        push!(js, j_mat)
                        push!(vals, val)
                        #N_test+= 1
                    end
                end
                if size(is)[1] > 0
                    Desc_Vec[i][d][R] = sparse(is, js, vals, Ne, Ne)
                end
            end
        end
        if verbosity > 1 && rank == 0
            println("Rank 0: Finished kernel features for mat Nr. ($i / $N_mats)")
        end
    end
    structure_descriptors = nothing
    GC.gc()
    #println("N_test",N_test)
    return Desc_Vec, (descr_sizes, N_dp)
end

function write_kernel_features_rankfile(
        Desc_Vec::Vector,
        meta::Tuple,
        filename_prefix::AbstractString,
        rank::Int,
        descr_dir::AbstractString)

    # ensure directory exists
    mkpath(descr_dir)

    # build filename
    filename = joinpath(descr_dir, "$(filename_prefix)_rank$(rank).h5")

    descr_sizes, N_dp = meta
    N_mats = length(Desc_Vec)

    h5open(filename, "w") do file

        # --- metadata ---
        meta_grp = create_group(file, "metadata")
        # convert tuple vector to 2×N numeric matrix
        descr_sizes_mat = reduce(hcat, [[a;b] for (a,b) in descr_sizes])
        meta_grp["descr_sizes"] = descr_sizes_mat
        meta_grp["N_dp"] = N_dp
        meta_grp["N_mats"] = N_mats

        # --- Desc_Vec storage ---
        data_grp = create_group(file, "Desc_Vec")

        # Parallel over N_mats using tforeach
        tforeach(1:N_mats) do i
            grp_i = create_group(data_grp, "mat_$i")

            for d in 1:N_dp
                grp_d = create_group(grp_i, "dp_$d")
                local_R = Desc_Vec[i][d]

                for R in eachindex(local_R)
                    spmat = local_R[R]
                    if nnz(spmat) > 0
                        I, J, V = findnz(spmat)
                        grp_R = create_group(grp_d, "R_$R")
                        grp_R["I"] = I
                        grp_R["J"] = J
                        grp_R["V"] = V
                        grp_R["m"] = size(spmat,1)
                        grp_R["n"] = size(spmat,2)
                    end
                end
            end
        end
    end

    return filename
end

# -----------------------------
# READ per-rank with tforeach
# -----------------------------
function read_kernel_features_rankfile(
        filename_prefix::AbstractString,
        rank::Int,
        descr_dir::AbstractString)

    filename = joinpath(descr_dir, "$(filename_prefix)_rank$(rank).h5")

    h5open(filename, "r") do file

        # --- metadata ---
        meta_grp = file["metadata"]
        descr_sizes_mat = read(meta_grp["descr_sizes"])
        N_dp = read(meta_grp["N_dp"])
        N_mats = read(meta_grp["N_mats"])

        descr_sizes = [(descr_sizes_mat[1,i], descr_sizes_mat[2,i])
                       for i in 1:size(descr_sizes_mat,2)]

        data_grp = file["Desc_Vec"]

        Desc_Vec = Vector{Vector{Vector{SparseMatrixCSC{Float64,Int}}}}(undef, N_mats)

        # Parallel over N_mats using tforeach
        tforeach(1:N_mats) do i
            N_R, Ne = descr_sizes[i]
            Desc_Vec[i] = Vector{Vector{SparseMatrixCSC{Float64,Int}}}(undef, N_dp)

            grp_i = data_grp["mat_$i"]

            for d in 1:N_dp
                Desc_Vec[i][d] = [spzeros(Float64, Ne, Ne) for _ in 1:N_R]
                grp_d = grp_i["dp_$d"]

                for R in 1:N_R
                    key = "R_$R"
                    if haskey(grp_d, key)
                        grp_R = grp_d[key]
                        I = read(grp_R["I"])
                        J = read(grp_R["J"])
                        V = read(grp_R["V"])
                        m = read(grp_R["m"])
                        n = read(grp_R["n"])
                        Desc_Vec[i][d][R] = sparse(I, J, V, m, n)
                    end
                end
            end
        end

        return Desc_Vec, (descr_sizes, N_dp)
    end
end



"""
    HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update, tol) -> HamiltonianKernel
"""
function HamiltonianKernel(params :: Vector{Float64},
    data_points,
    sim_params,
    structure_descriptors,
    update :: Bool,
    tol :: Float64;
    conf = get_empty_config(),
    rank = 0
    )
    calc_desc_mode = get_get_desc_mode(conf)
    if calc_desc_mode == "read"
        @info "Reading kernel features from file: $rank"
        feature_vec, feature_shape = read_kernel_features_rankfile("descr", rank,"descr")
    else
        feature_vec, feature_shape = get_kernel_features(structure_descriptors, data_points, sim_params, tol, conf = conf, rank = rank)
    end
    if calc_desc_mode == "write"
        @info "Writing kernel features to file: $rank"
        write_kernel_features_rankfile(feature_vec, feature_shape,"descr",  rank, "descr")
    end
    return HamiltonianKernel(params,data_points, sim_params, update, feature_vec, feature_shape)
end

"""
    HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update) -> HamiltonianKernel
"""
function HamiltonianKernel(params :: Vector{Float64},
    data_points,
    sim_params,
    structure_descriptors,
    update :: Bool;
    conf = get_empty_config(),
    rank = 0
    )

    return HamiltonianKernel(params,
    data_points,
    sim_params,
    structure_descriptors,
    update,
    1e-8;
    conf = conf,
    rank = rank)
end

#ham_val = EffectiveHamiltonian(val_strcs, val_bases, comm_active, conf, rank=active_rank, nranks=active_size, ml_data_points=get_ml_data_points(ham_train, conf))

"""
    HamiltonianKernel(strcs, bases, model, conf)

Constructor for a HamiltonianKernel model.
"""
function HamiltonianKernel(strcs::Vector{<:Structure}, bases::Vector{<:Basis}, model, comm, conf=get_empty_config(), data_points = nothing; 
                            verbosity=get_verbosity(conf),
                            Ncluster=get_ml_ncluster(conf),
                            Npoints=get_ml_npoints(conf),
                            sim_params=get_sim_params(conf), 
                            sp_tol=get_sp_tol(conf),
                            update_ml=get_ml_update(conf),
                            sample_strat = get_sample_strat(conf),
                            rank=0,
                            nranks=1)
    structure_descriptors = Vector{Any}(undef, length(strcs))
    tmap!(structure_descriptors, eachindex(strcs)) do n
        get_tb_descriptor(
            model.hs[n],
            model.params,
            strcs[n],
            bases[n],
            conf
        )
    end
    strcs = nothing
    bases = nothing
    GC.gc()
    if get_ml_init_params(conf)[1] ∈ ['r', 'z', 'o'] && data_points === nothing
        Npoints_local = floor(Int64, Npoints / nranks)
        if sample_strat == "cluster"
            @info "Sampling data points using clustering strategy with Ncluster = $Ncluster"
            data_points_local = sample_structure_descriptors(reshape_structure_descriptors(structure_descriptors), Ncluster=Ncluster, Npoints=Npoints_local, ml_sampling=get_ml_sampling(conf))
        else
            data_points_local = sample_structure_descriptors_random(reshape_structure_descriptors(structure_descriptors), Npoints=Npoints_local)
            @info "Sampling data points using random strategy with Npoints_local = $Npoints_local"
        end

        local_counts::Int32 = length(data_points_local)
        counts = MPI.Gather(local_counts, 0, comm)
        counts = MPI.bcast(counts, 0, comm)

        data_points_buf = nothing
        if rank == 0
            data_points_buf = MPI.VBuffer(similar(data_points_local, sum(counts)), counts)
            println("Npoints_local sampled: ", Npoints_local)
            println("nranks: ", nranks)
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
    elseif data_points === nothing
        _, data_points = read_ml_params(conf, filename=get_ml_init_params(conf))
    else
        
    end
    params, data_points = init_ml_params!(data_points, conf)

    return HamiltonianKernel(params, data_points, sim_params,structure_descriptors, update_ml, sp_tol;
    conf =conf,
    rank = rank)
end




exp_sim(x₁, x₂; σ=√0.05)::Float64 = exp(-normdiff(x₁, x₂)^2 / (2σ^2))






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


function get_hr(kernel::HamiltonianKernel, mode, index; apply_soc=false)
    @views desc_vec = kernel.feature_vec[index]
    N_dp = kernel.feature_shape[2]
    (NR, Ne) = kernel.feature_shape[1][index]
    Hr = get_empty_complex_hamiltonians(Ne, NR, mode)

    Hr = tmapreduce(.+, 1:N_dp) do d
        desc_vec[d] .* kernel.params[d]
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

function write_datapoints(data_points::Vector{SVector{8, Float64}}, target_dir::String, conf=get_empty_config(); filename=get_ml_filename(conf))
    open(joinpath(target_dir, filename*".dat"), "w") do file
        # Write header to file
        #params = init_ml_params!(data_points, conf)[1]
        Nparams = length(data_points)
        params = zeros(Nparams)
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_sim_params(conf))
        println(file, "  env_scale = ", get_env_scale(conf))
        println(file, "  apply_distortion = ", get_apply_distortion(conf))
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
    read_ml_params(conf=get_empty_config(); filename=get_ml_filename(conf))

Reads the parameters for a HamiltonianKernel model from a file and returns the parameters and associated data points.

# Arguments
- `conf`: A configuration object (default: `get_empty_config()`) containing simulation parameters and settings.
- `filename`: The name of the `.dat` file to read from (default: `get_ml_filename(conf)`).
"""
function read_ml_params(target_dir::String, conf=get_empty_config(); filename=get_ml_filename(conf))
    if !occursin(".dat", filename); filename *= ".dat"; end
    lines = open_and_read(joinpath(target_dir, filename))
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
function read_ml_params( conf=get_empty_config(); filename=get_ml_filename(conf))
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

function get_model_gradient(kernel::HamiltonianKernel, indices, reg, dL_dHr; soc=false)
    dparams = zeros(length(kernel.params))
    if kernel.update
        tforeach( eachindex(dparams)) do n
            for (bi, index) in enumerate(indices)
                @views desc_vec = kernel.feature_vec[index][n]
                for R in eachindex(dL_dHr[bi])
                    for (i, j, exp_val) in zip(findnz(desc_vec[R])...)
                        if !soc
                            dparams[n] += exp_val .* real(dL_dHr[bi][R][i, j])
                        else
                            i1 = 2*i-1; j1 = 2*j-1
                            i2 = 2*i; j2 = 2*j
                            dparams[n] += exp_val .* real(dL_dHr[bi][R][i1, j1] + dL_dHr[bi][R][i2, j2])
                        end
                    end
                end
            end
        end
        dparams_penal = backward(reg, kernel.params)
        return dparams .+ dparams_penal
    else 
        return dparams
    end
end

