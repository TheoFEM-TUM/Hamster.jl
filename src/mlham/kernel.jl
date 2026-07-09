"""
    mutable struct HamiltonianKernel{T1, T2}

A kernel structure used for computing weighted similarity functions.

# Fields
- `ws :: Vector{Float64}`: Weights for each sample point.
- `xs :: Vector{T1}`: Sample points.
- `sim_params :: T2`: Parameters for the similarity function.
"""
mutable struct HamiltonianKernel{}
    params :: Vector{Float64}
    kp :: Kernelpoints
    sim_params :: Float64
    update :: Bool
    sm :: SimMat
end



"""
    HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update, tol) -> HamiltonianKernel
"""
function HamiltonianKernel(params :: Vector{Float64},
    kp,
    sim_params,
    structure_descriptors,
    update :: Bool,
    tol :: Float64;
    conf = get_empty_config(),
    rank = 0,
    systems = nothing
    )
    sm = get_kernel_features(structure_descriptors, kp, sim_params, tol, conf = conf, rank = rank, systems = systems)

    return HamiltonianKernel(params,kp, sim_params, update, sm)
end

"""
    HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update) -> HamiltonianKernel
"""
function HamiltonianKernel(params :: Vector{Float64},
    kp,
    sim_params,
    structure_descriptors,
    update :: Bool,
    weights;
    conf = get_empty_config(),
    rank = 0
    )

    return HamiltonianKernel(params,
    kp,
    sim_params,
    structure_descriptors,
    update,
    1e-8,
    weights;
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
                            only_sample = get_only_sample(conf),
                            kernel_chunck_size = get_kernel_chunk_size(conf),
                            rank=0,
                            nranks=1)
    Nstrc = length(strcs)
    structure_descriptors = Vector{Any}(undef, Nstrc)
    tmap!(structure_descriptors, 1:Nstrc) do n
        get_tb_descriptor(
            model.hs[n],
            model.params,
            strcs[n],
            bases[n],
            conf
        )
    end
    systems = [strc.system for strc in strcs]
    strcs = nothing
    bases = nothing
    GC.gc()
    if get_ml_init_params(conf)[1] ∈ ['r', 'z', 'o'] && data_points === nothing
        Npoints_local = floor(Int64, Npoints / nranks)
        Ncluster_local = floor(Int64, Ncluster / nranks)
        #Ncluster_local = Ncluster
        Np_per_strc = zeros(Int, length(structure_descriptors))
        if sample_strat == "cluster"
            @info "Sampling data points using clustering strategy with Ncluster = $Ncluster_local"
            
            reshaped_descr = reshape_structure_descriptors(structure_descriptors, Np_per_strc, get_weight_factor(conf))
            data_points_local = sample_structure_descriptors(reshaped_descr, Ncluster=Ncluster_local, Npoints=Npoints_local, ml_sampling=get_ml_sampling(conf), dim_weights=get_dim_weights(conf))
            #println("Np_per_strc: ", Np_per_strc)
        elseif sample_strat == "single_rank"
            @info "Sampling data points using single_rank strategy with Ncluster = $Ncluster"

            # build local descriptor matrix
            data_points_local = reshape_structure_descriptors(structure_descriptors, Np_per_strc, get_weight_factor(conf))

            #println("local size: ", size(data_points_local))

            dim = size(data_points_local, 1)
            local_cols = size(data_points_local, 2)

            # gather number of columns from all ranks
            counts_1 = MPI.Gather(local_cols, 0, comm)
            counts_1 = MPI.bcast(counts_1, 0, comm)

            data_points_buf = nothing
            recv_mat = nothing

            if rank == 0
                total_cols = sum(counts_1)

                println("total columns: ", total_cols)

                recv_mat = Matrix{Float32}(undef, dim, total_cols)

                # MPI expects number of elements
                counts_elements = counts_1 .* dim

                data_points_buf = MPI.VBuffer(vec(recv_mat), counts_elements)
            end

            # gather all matrices as flat vectors
            MPI.Gatherv!(
                vec(data_points_local),
                data_points_buf,
                0,
                comm
            )

            # gather structure counts
            #Np_per_strc_per_rank = MPI.Gather(Np_per_strc, 0, comm)

            if rank == 0
                total_cols = sum(counts_1)

                reshaped_descr = reshape(data_points_buf.data, dim, total_cols)

                println("size reshaped_descr: ", size(reshaped_descr))

                # combine structure counts
                #Np_per_strc = vcat(Np_per_strc_per_rank...)

                #println("Np_per_strc: ", Np_per_strc)

                # sampling
                data_points_local = sample_structure_descriptors(
                    reshaped_descr,
                    Ncluster = Ncluster,
                    Npoints = Npoints,
                    ml_sampling = get_ml_sampling(conf),
                    dim_weights = get_dim_weights(conf)
                )

            else
                data_points_local = Matrix{Float64}(undef, dim - 1, 0)
            end

        elseif sample_strat == "cluster_single"
            
            data_points_local = Vector{Any}(undef, Nstrc)
            N_points_vec = [1 for i in 1:Nstrc]
            tforeach(1:Nstrc) do i
                strc_descriptors = reshape_structure_descriptor_single_system(structure_descriptors[i])
                N_descr = size(strc_descriptors)[2]
                N_points_single = ceil(Int, N_descr/4)
                Ncluster_single = ceil(Int, N_points_single/4)
                #N_points_single = Npoints
                #Ncluster_single = Ncluster
                N_points_vec[i] = N_points_single
                data_points_local[i] = sample_structure_descriptors(strc_descriptors, Ncluster=Ncluster_single, Npoints=N_points_single, ml_sampling=get_ml_sampling(conf), dim_weights=get_dim_weights(conf))
                #println(size(data_points_local[i]))
                system = systems[i]
                @info "$system Sampling data points using clustering single strategy with Ncluster = $Ncluster_single and Npoints_local_total = $N_points_single"
            end
            data_points_local = reduce(vcat, data_points_local)
        elseif sample_strat == "split"
            if rank == 0
                @info "Sampling data points using split strategy with Ncluster = $Ncluster"
            end

            # build local descriptor matrix
            data_points_local = reshape_structure_descriptors(structure_descriptors, Np_per_strc, get_weight_factor(conf))

            #println("local size: ", size(data_points_local))

            dim = size(data_points_local, 1)
            local_cols = size(data_points_local, 2)

            # gather number of columns from all ranks
            counts_1 = MPI.Gather(local_cols, 0, comm)
            counts_1 = MPI.bcast(counts_1, 0, comm)

            data_points_buf = nothing
            recv_mat = nothing

            if rank == 0
                total_cols = sum(counts_1)

                println("total columns: ", total_cols)

                recv_mat = Matrix{Float32}(undef, dim, total_cols)

                # MPI expects number of elements
                counts_elements = counts_1 .* dim

                data_points_buf = MPI.VBuffer(vec(recv_mat), counts_elements)
            end

            # gather all matrices as flat vectors
            MPI.Gatherv!(
                vec(data_points_local),
                data_points_buf,
                0,
                comm
            )

            # gather structure counts
            #Np_per_strc_per_rank = MPI.Gather(Np_per_strc, 0, comm)

            if rank == 0
                time = @elapsed begin
                    total_cols = sum(counts_1)

                    reshaped_descr = reshape(data_points_buf.data, dim, total_cols)

                    println("size reshaped_descr: ", size(reshaped_descr))


                    # build local descriptor matrix

                    sub_descr = build_submatrices(reshaped_descr, conf)
                    keys_list = []
                    for key in keys(sub_descr)
                        push!(keys_list, key)
                        #println(keys_list[end])
                    end

                    N_key = length(keys_list)

                    Np_Nc_dict = calc_npoint_ncluster(sub_descr, Npoints, Ncluster, conf)

                    sort!(keys_list; by = k -> begin
                        Np, Nc = Np_Nc_dict[k]
                        Np * Nc      # or Nc^2, or whatever best predicts runtime
                    end, rev = true)
                    
                    data_points_local = Vector{Any}(undef, N_key)

                    N_key_finished = Threads.Atomic{Int}(0)

                    tmap!(data_points_local, 1:N_key) do n
                        Np, Nc = Np_Nc_dict[keys_list[n]]

                        result = nothing
                        elapsed = @elapsed begin
                            result = sample_structure_descriptors(
                                sub_descr[keys_list[n]],
                                Ncluster=Nc,
                                Npoints=Np,
                                ml_sampling=get_ml_sampling(conf),
                                dim_weights=get_dim_weights(conf)
                            )
                        end

                        finished = Threads.atomic_add!(N_key_finished, 1) + 1

                        @info "Thread $(Threads.threadid()) finished key $(keys_list[n]) with Nc=$Nc in $(round(elapsed, digits=2)) s. ($finished / $N_key)"

                        result
                    end
                    
                    data_points_local = reduce(vcat, data_points_local)
                end
                println("Sampling data points using split strategy finished in $time s.")
            else
                data_points_local = Matrix{Float64}(undef, dim - 1, 0)
            end
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
        weights = nothing
        if rank == 0
            d_counts = Hamster.countmap(map(x -> round.(x; digits=6), data_points))
            data_points = collect(keys(d_counts))
            weights = collect(values(d_counts))
            #data_points = data_points_buf.data
        end
        data_points = MPI.bcast(data_points, comm)
        weights = MPI.bcast(weights, comm)

        N_real = length(weights)
        N_theo = sum(weights)
        # COV_EXCL_START
        if N_real ≠ Npoints && rank == 0 && verbosity > 0
            #@info "Number of samples changed from $Npoints to $N_theo"
        end
        if rank == 0 && verbosity > 0
            @info "Number of datapoints changed from $N_theo to $N_real"
        end

    elseif data_points === nothing
        _, data_points = read_ml_params(conf, filename=get_ml_init_params(conf))
        weights = ones(Int, length(data_points))
    else
        weights = ones(Int, length(data_points))
    end
    params, data_points = init_ml_params!(data_points, conf)
    kp, params = get_sorted_Kernelpoints(data_points, weights, params, conf)
    
    if get_verbosity(conf) >= 2
        #ok = check_consistency(kp.datapoints, params; key_ranges=kp.key_ranges)
        #ok2 = verify_kernelpoints(kp, conf)
    end


    if rank == 0
        filename = only_sample ? "ml_params_sample" : "ml_params_temp"
        write_params((params, data_points), conf, filename=filename)
        if only_sample
            @info "Only sampling data points and writing to file because only_sample=true. Stopping program now."
            sleep(10)
            throw(ErrorException("Forcefully stopping programm because of only_sample=true"))
        end
    end

    return HamiltonianKernel(params, kp, sim_params,structure_descriptors, update_ml, sp_tol;
            conf =conf,
            rank = rank,
            systems = systems)



end


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
    data_points = kernel.kp.datapoints
    open(filename*".dat", "w") do file
        # Write header to file
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_sim_params(conf))
        println(file, "  env_scale = ", get_env_scale(conf))
        println(file, "  Z_scale = ", get_Z_scale(conf))
        println(file, "  overlap_scale = ", get_overlap_scale(conf))
        println(file, "  strc_scale = ", get_strc_scale(conf))
        println(file, "  R_scale = ", get_R_scale(conf))
        println(file, "  apply_distortion = ", get_apply_distortion(conf))
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

function write_params(params_data_points_tuple::Tuple{Vector{Float64}, Vector{SVector{9, Float64}}}, conf=get_empty_config(); filename=get_ml_filename(conf))
    open(filename*".dat", "w") do file
        # Write header to file
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_sim_params(conf))
        println(file, "  env_scale = ", get_env_scale(conf))
        println(file, "  Z_scale = ", get_Z_scale(conf))
        println(file, "  overlap_scale = ", get_overlap_scale(conf))
        println(file, "  strc_scale = ", get_strc_scale(conf))
        println(file, "  R_scale = ", get_R_scale(conf))
        println(file, "  apply_distortion = ", get_apply_distortion(conf))
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

function write_datapoints(data_points::Vector{SVector{9, Float64}}, target_dir::String, conf=get_empty_config(); filename=get_ml_filename(conf))
    open(joinpath(target_dir, filename*".dat"), "w") do file
        # Write header to file
        #params = init_ml_params!(data_points, conf)[1]
        Nparams = length(data_points)
        params = zeros(Nparams)
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_sim_params(conf))
        println(file, "  env_scale = ", get_env_scale(conf))
        println(file, "  Z_scale = ", get_Z_scale(conf))
        println(file, "  overlap_scale = ", get_overlap_scale(conf))
        println(file, "  strc_scale = ", get_strc_scale(conf))
        println(file, "  R_scale = ", get_R_scale(conf))
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
    N = length(lines[12]) - 1
    warn_yes = get_hyperopt_niter(conf) < 2
    # Check that header params match Config
    checks = [
        ("ml_rcut", get_ml_rcut(conf), 2),
        ("sim_params", get_sim_params(conf), 3),
        ("env_scale", get_env_scale(conf), 4),
        ("Z_scale", get_Z_scale(conf), 5),
        ("overlap_scale", get_overlap_scale(conf), 6),
        ("strc_scale", get_strc_scale(conf), 7),
        ("R_scale", get_R_scale(conf), 8),
    ]

    for (name, expected, line) in checks
        val = parse(Float64, lines[line][end])
        if val != expected && warn_yes
            @warn "$name mismatch" parsed=val expected=expected
        end
    end

    val = parse(Bool, lines[9][end])
    if val != get_apply_distortion(conf) && warn_yes
        @warn "apply_distortion mismatch" parsed=val expected=get_apply_distortion(conf)
    end

    data_points = SVector{N, Float64}[]
    params = Float64[]
    for line in lines[12:end]
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
    N = length(lines[12]) - 1
    warn_yes = get_hyperopt_niter(conf) < 2
    # Check that header params match Config
    checks = [
        ("ml_rcut", get_ml_rcut(conf), 2),
        ("sim_params", get_sim_params(conf), 3),
        ("env_scale", get_env_scale(conf), 4),
        ("Z_scale", get_Z_scale(conf), 5),
        ("overlap_scale", get_overlap_scale(conf), 6),
        ("strc_scale", get_strc_scale(conf), 7),
        ("R_scale", get_R_scale(conf), 8),
    ]

    for (name, expected, line) in checks
        val = parse(Float64, lines[line][end])
        if val != expected && warn_yes
            @warn "$name mismatch" parsed=val expected=expected
        end
    end

    val = parse(Bool, lines[9][end])
    if val != get_apply_distortion(conf) && warn_yes
        @warn "apply_distortion mismatch" parsed=val expected=get_apply_distortion(conf)
    end
    data_points = SVector{N, Float64}[]
    params = Float64[]
    for line in lines[12:end]
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
    #throw_error = length(kernel.data_points) ≠ length(params)
    #if throw_error
    #    error("Parameter vector is not of correct size ($(length(kernel.data_points)) ≠ $(length(params)))!")
    #else
        kernel.params = params
    #end
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

function get_model_gradient_old(kernel::HamiltonianKernel, indices, reg, dL_dHr; soc=false)
    dparams = zeros(length(kernel.params))
    weights = kernel.kp.weights
    #weights = ones(length(dparams)) # for unweighted gradients
    #weights = (weights .-1) .*2 .+1
    if kernel.update
        tforeach( eachindex(dparams)) do n
            for (bi, index) in enumerate(indices)
                @views desc_vec = kernel.sm.feature_vec[index][n]
                for R in eachindex(dL_dHr[bi])
                    for (i, j, exp_val) in zip(findnz(desc_vec[R])...)
                        if !soc
                            dparams[n] += weights[n] .* exp_val .* real(dL_dHr[bi][R][i, j])
                        else
                            i1 = 2*i-1; j1 = 2*j-1
                            i2 = 2*i; j2 = 2*j
                            dparams[n] += weights[n] .* exp_val .* real(dL_dHr[bi][R][i1, j1] + dL_dHr[bi][R][i2, j2])
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