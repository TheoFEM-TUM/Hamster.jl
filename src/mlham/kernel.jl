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
    key_ranges :: Dict{Tuple{Int,Int,Int},UnitRange{Int}}
    sim_params :: Float64
    update :: Bool
    feature_vec :: Vector{T3}
    feature_shape :: Tuple{Vector{T2}, Int64}
    weights :: Vector{Int64}
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

function get_kernel_features_old(structure_descriptors, data_points, sim_params, tol = 1e-8; conf = get_empty_config(), rank = 0, systems = nothing)
    #verbosity = get_verbosity(conf)
    tol = 0.5
    #println(tol)
    #println("NTHREADS",Threads.nthreads())
    N_mats = size(structure_descriptors)[1]
    systems = systems === nothing ? [string("system_", i) for i in 1:N_mats] : systems
    N_dp = size(data_points)[1]
    descr_sizes = [(size(structure_descriptors[i])[1], size(structure_descriptors[i][1])[1]) for i in 1:N_mats]
    Desc_Vec = [ [[ spzeros(Float64, descr_sizes[i][2], descr_sizes[i][2]) for _ in 1:descr_sizes[i][1] ] for d in 1:N_dp]
      for i in 1:N_mats ]
    N_test = zeros(Int, N_mats)
    N_max = 0
    for i in 1:N_mats
    #for i in 1:N_mats
        @views h_env = structure_descriptors[i]
        #for d in 1:N_dp
        tforeach(1:N_dp) do d
            @views data_point = data_points[d]
            N_R, Ne = descr_sizes[i]
            for R in 1:N_R
                is = Vector{Int32}() 
                js = Vector{Int32}() 
                vals = Vector{Float64}() 
                for (i_mat, j_mat, hin) in zip(findnz(h_env[R])...)
                    val = exp_sim(data_point, hin, σ=sim_params)
                    #N_max += 1
                    if abs(val) > tol
                        push!(is, i_mat)
                        push!(js, j_mat)
                        push!(vals, val)
                        N_test[i]+= 1
                    end
                end
                if size(is)[1] > 0
                    Desc_Vec[i][d][R] = sparse(is, js, vals, Ne, Ne)
                end
            end
        end
        @info "Rank $rank: Finished kernel features for mat $(systems[i]) Nr. ($i / $N_mats) with Npoints = $(N_test[i])"
    end
    structure_descriptors = nothing
    GC.gc()
    #println("N_test",N_test)
    return Desc_Vec, (descr_sizes, N_dp)
end


function get_kernel_features(structure_descriptors, data_points, key_ranges, sim_params, tol = 1e-8; conf = get_empty_config(), rank = 0, systems = nothing)
    #verbosity = get_verbosity(conf)
    #data_points_dict = build_submatrices(data_points, conf)
    Z_scale=get_Z_scale(conf)
    overlap_scale=get_overlap_scale(conf)
    tol = 0.5
    #println(tol)
    #println("NTHREADS",Threads.nthreads())
    N_mats = size(structure_descriptors)[1]
    systems = systems === nothing ? [string("system_", i) for i in 1:N_mats] : systems
    N_dp = size(data_points)[1]
    #dim = size(data_points[1])[1]
    descr_sizes = [(size(structure_descriptors[i])[1], size(structure_descriptors[i][1])[1]) for i in 1:N_mats]

    Desc_Vec = [
        [ begin
            n = nnz(structure_descriptors[i][R])
            (Vector{Tuple{SparseVector{Float64, Int64}, Tuple{Int64, Int64, Tuple{Int64,Int64,Int64}}}}(undef, n), n)
        end
        for R in 1:descr_sizes[i][1] ]
        for i in 1:N_mats
    ]
    N_test = zeros(Int64, N_mats)
    N_total = zeros(Int64, N_mats)
    for i in 1:N_mats
        N_R, Ne = descr_sizes[i]
        for R in 1:N_R
            h_env_R = structure_descriptors[i][R]
            #Nnz = Desc_Vec[i][R][2]
            touples = collect(zip(findnz(h_env_R)...))
            N_total_temp  = length(touples)
            N_test_temp = zeros(N_total_temp)
            tforeach(1:length(touples)) do n
                i_mat, j_mat, hin = touples[n]
                overlap_id = floor(Int, hin[1] / overlap_scale)
                Z_1_id = floor(Int, hin[2] / Z_scale)
                Z_2_id = floor(Int, hin[3] / Z_scale)
                key = (overlap_id, Z_1_id, Z_2_id)
                #data_points_mat = reduce(hcat, @view data_points[key_ranges[key]])
                data_points_vector = @view data_points[key_ranges[key]]
                #N_dp = size(data_points_mat)[2]
                val_vec = exp_sim_all(data_points_vector, hin, σ=sim_params)
                val_vec[abs.(val_vec) .<= tol] .= 0
                val_vec = sparse(val_vec)
                covered = nnz(val_vec) > 0 ? 1 : 0
                N_test_temp[n] = covered
                #N_total[i] += 1
                #println(nnz(val_vec))
                Desc_Vec[i][R][1][n] = (val_vec,(i_mat,j_mat, key))
            end
            N_test[i] += sum(N_test_temp)
            N_total[i] += N_total_temp
        end
        @info "Rank $rank: Finished kernel features for mat $(systems[i]) Nr. ($i / $N_mats) with Ncovered = ( $(N_test[i]) / $(N_total[i]) ) || $(ceil(Int, N_test[i] / N_total[i] * 100 )) %"
    end
    structure_descriptors = nothing
    GC.gc()
    #println("N_test",N_test)
    return Desc_Vec, (descr_sizes, N_dp)
end



"""
    HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update, tol) -> HamiltonianKernel
"""
function HamiltonianKernel(params :: Vector{Float64},
    data_points,
    key_ranges,
    sim_params,
    structure_descriptors,
    update :: Bool,
    tol :: Float64,
    weights;
    conf = get_empty_config(),
    rank = 0,
    systems = nothing
    )

    feature_vec, feature_shape = get_kernel_features(structure_descriptors, data_points,key_ranges, sim_params, tol, conf = conf, rank = rank, systems = systems)
    #data_points = Vector{typeof(data_points[1])}(undef, 0)

    return HamiltonianKernel(params,data_points,key_ranges, sim_params, update, feature_vec, feature_shape, weights)
end

"""
    HamiltonianKernel(params, data_points, sim_params, structure_descriptors, update) -> HamiltonianKernel
"""
function HamiltonianKernel(params :: Vector{Float64},
    data_points,
    key_ranges ,
    sim_params,
    structure_descriptors,
    update :: Bool,
    weights;
    conf = get_empty_config(),
    rank = 0
    )

    return HamiltonianKernel(params,
    data_points,
    key_ranges,
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
            data_points_local = sample_structure_descriptors(reshaped_descr, Ncluster=Ncluster_local, Npoints=Npoints_local, ml_sampling=get_ml_sampling(conf))
            #println("Np_per_strc: ", Np_per_strc)
        elseif sample_strat == "single_rank"
            @info "Sampling data points using single_rank strategy with Ncluster = $Ncluster"

            # build local descriptor matrix
            data_points_local = reshape_structure_descriptors(structure_descriptors, Np_per_strc, get_weight_factor(conf))

            println("local size: ", size(data_points_local))

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
                    ml_sampling = get_ml_sampling(conf)
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
                data_points_local[i] = sample_structure_descriptors(strc_descriptors, Ncluster=Ncluster_single, Npoints=N_points_single, ml_sampling=get_ml_sampling(conf))
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

            println("local size: ", size(data_points_local))

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

                    data_points_local = Vector{Any}(undef, N_key)
                    tmap!(data_points_local, 1:N_key) do n
                        Np, Nc = Np_Nc_dict[keys_list[n]]
                        #println("Sampling key: ", keys_list[n], " with Np = ", Np, " and Nc = ", Nc)
                        sample_structure_descriptors(sub_descr[keys_list[n]], Ncluster=Nc, Npoints=Np, ml_sampling=get_ml_sampling(conf))      
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

    data_points, params, key_ranges = sort_by_key(data_points, params, conf)
    ok = check_consistency(data_points, params; key_ranges=key_ranges)


    if rank == 0
        filename = only_sample ? "ml_params_sample" : "ml_params_temp"
        write_params((params, data_points), conf, filename=filename)
        if only_sample
            @info "Only sampling data points and writing to file because only_sample=true. Stopping program now."
            sleep(10)
            throw(ErrorException("Forcefully stopping programm because of only_sample=true"))
        end
    end

    return HamiltonianKernel(params, data_points, key_ranges, sim_params,structure_descriptors, update_ml, sp_tol, weights;
            conf =conf,
            rank = rank,
            systems = systems)



end




exp_sim2(x1::SVector{9,Float64}, x2::SVector{9,Float64}; σ=√0.05) =
    exp(-normdiff(x1, x2)^2 / (2σ^2))

exp_sim_all(x1, x2::SVector{9,Float64}; σ=√0.05) = exp_sim2.(x1, Ref(x2); σ=σ)




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
    (NR, Ne) = kernel.feature_shape[1][index]
    key_ranges = kernel.key_ranges
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
    data_points = kernel.data_points
    open(filename*".dat", "w") do file
        # Write header to file
        println(file, "begin ", get_system(conf))
        println(file, "  rcut = ", get_ml_rcut(conf))
        println(file, "  sim_params = ", get_sim_params(conf))
        println(file, "  env_scale = ", get_env_scale(conf))
        println(file, "  Z_scale = ", get_Z_scale(conf))
        println(file, "  overlap_scale = ", get_overlap_scale(conf))
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
    N = length(lines[10]) - 1

    # Check that header params match Config
    @assert parse(Float64, lines[2][end]) == get_ml_rcut(conf)
    @assert parse(Float64, lines[3][end]) == get_sim_params(conf)
    @assert parse(Float64, lines[4][end]) == get_env_scale(conf)
    @assert parse(Bool, lines[7][end]) == get_apply_distortion(conf)

    data_points = SVector{N, Float64}[]
    params = Float64[]
    for line in lines[10:end]
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
    N = length(lines[10]) - 1

    # Check that header params match Config
    @assert parse(Float64, lines[2][end]) == get_ml_rcut(conf)
    @assert parse(Float64, lines[3][end]) == get_sim_params(conf)
    @assert parse(Float64, lines[4][end]) == get_env_scale(conf)
    @assert parse(Bool, lines[7][end]) == get_apply_distortion(conf)

    data_points = SVector{N, Float64}[]
    params = Float64[]
    for line in lines[10:end]
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
    key_ranges = kernel.key_ranges
    if kernel.update
        for (bi, index) in enumerate(indices)
            
            for R in eachindex(dL_dHr[bi])
                @views desc_vec, nnz_ham = kernel.feature_vec[index][R]
                tforeach(1:nnz_ham) do m
                    tid = Threads.threadid()
                    @views desc_vec_small, (i,j,key) =  desc_vec[m]

                    if !soc
                        dparams_threads[tid][key_ranges[key]]  .+= desc_vec_small .* real(dL_dHr[bi][R][i, j])
                    else
                        i1 = 2*i-1; j1 = 2*j-1
                        i2 = 2*i; j2 = 2*j
                        dparams_threads[tid][key_ranges[key]] .+= desc_vec_small .* real(dL_dHr[bi][R][i1, j1] + dL_dHr[bi][R][i2, j2])
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
    weights = kernel.weights
    #weights = ones(length(dparams)) # for unweighted gradients
    #weights = (weights .-1) .*2 .+1
    if kernel.update
        tforeach( eachindex(dparams)) do n
            for (bi, index) in enumerate(indices)
                @views desc_vec = kernel.feature_vec[index][n]
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