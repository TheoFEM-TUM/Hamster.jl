# Active learning: fits alternate with the selection of new training structures from a pool of labeled structures
# (and, optionally, new kernel points)

"""
    run_calculation(::Val{:active_learning}, comm, conf; rank=0, nranks=1)

Active learning with a pool of structures with known eigenvalues (one MD system). The training set starts with `Nconf`
random structures; `nval` validation structures are drawn once and never enter the training set; all other structures
in `Nconf_min:Nconf_max` form the pool. In every round:

1. The models are fitted to the current training set (see `Optimizer`), starting from the parameters of the previous
   round. The parameters are written to `al/round_<n>/`.
2. `nadd` structures are moved from the pool to the training set: with `selection = coverage` the structures whose
   descriptors are covered worst by the kernel points (lowest mean similarity to the most similar point), with
   `selection = random` random structures (for comparison).
3. With `npoints_add > 0`, up to `npoints_add` descriptors of the new structures become kernel points (weight 0), by
   farthest-point sampling starting from the existing points; only descriptors with a similarity below `add_below` to
   all points are added.

After the last round, `params.dat` and `ml_params.dat` of the round with the lowest validation loss are copied to the
run folder, and a summary of all rounds is written to `hamster.out` and `al/history.h5`.
"""
function run_calculation(::Val{:active_learning}, comm, conf::Config; rank=0, nranks=1, verbosity=get_verbosity(conf),
        nrounds=get_activelearning_nrounds(conf), nadd=get_activelearning_nadd(conf), nval=get_activelearning_nval(conf),
        selection=lowercase(get_activelearning_selection(conf)), npoints_add=get_activelearning_npoints_add(conf),
        add_below=get_activelearning_add_below(conf))

    systems = get_systems(conf)
    (length(systems) == 1 && haskey(conf, "Supercell")) || error("Active learning needs one MD system (Supercell block).")
    get_validate(conf) || error("Active learning needs a validation set: set val_mode = md in the Optimizer block.")
    selection ∈ ("coverage", "random") || error("Unknown selection '$selection' (coverage or random).")
    get_ml_model(conf) || selection == "random" || error("selection = coverage needs an ML model.")
    get_ml_sim_mat(conf) && error("Active learning does not support sim_mat = true.")
    system = systems[1]

    # Initial training set, fixed validation set and pool (disjoint)
    candidates = collect(get_Nconf_min(conf):get_Nconf_max(conf))
    train, val = Int64[], Int64[]
    if rank == 0
        train = sort(sample(candidates, get_Nconf(conf), replace=false))
        val = sort(sample(setdiff(candidates, train), nval, replace=false))
    end
    train, val = MPI.bcast(train, comm, root=0), MPI.bcast(val, comm, root=0)

    prof_al = HamsterProfiler(1, conf, Niter=nrounds, Nbatch=1)
    history = Dict("Ntrain" => Int64[], "Npoints" => Int64[], "L_train" => Float64[], "L_val" => Float64[])
    round_trains = Vector{Vector{Int64}}()
    init_dir = ""
    for round in 1:nrounds
        dir = joinpath("al", "round_$round")
        if rank == 0
            mkpath(dir)
            h5open(joinpath(dir, "config_inds.h5"), "w") do file
                file["train_config_inds"] = train
                file["val_config_inds"] = val
            end
            verbosity > 0 && println("Active learning round $round of $nrounds: $(length(train)) training structures")
        end
        MPI.Barrier(comm)
        push!(round_trains, copy(train))

        conf_round = deepcopy(conf)
        set_value!(conf_round, "config_inds", "Supercell", joinpath(dir, "config_inds.h5"))
        set_value!(conf_round, "val_config_inds", "Supercell", joinpath(dir, "config_inds.h5"))
        set_value!(conf_round, "Nconf", "Supercell", length(train))
        isempty(init_dir) || init_from_files!(conf_round, joinpath(init_dir, "params.dat"), joinpath(init_dir, "ml_params_next.dat"))

        prof, ham = fit_model(comm, conf_round, rank=rank, nranks=nranks, write_output=false)
        data_points = get_ml_model(conf) ? MPI.bcast(rank == 0 ? get_ml_data_points(ham, conf) : nothing, comm, root=0) : []
        if rank == 0
            write_params_atomically(ham, conf_round, dir=dir)
            L_val = minimum(filter(>(0), prof.L_val), init=Inf)
            L_train = mean(prof.L_train[:, end])
            push!(history["Ntrain"], length(train)); push!(history["Npoints"], length(data_points))
            push!(history["L_train"], L_train); push!(history["L_val"], L_val)
            prof_al.L_train[1, round] = L_train; prof_al.L_val[round] = L_val
            verbosity > 0 && @printf("Round %d: training loss %.5f, best validation loss %.5f, %d kernel points\n",
                                     round, L_train, L_val, length(data_points))
        end
        round == nrounds && break

        # New training structures (and kernel points) from the pool
        pool = setdiff(candidates, train, val)
        n = min(nadd, length(pool))
        n == 0 && break
        new_points = similar(data_points, 0)
        if selection == "coverage" || npoints_add > 0
            scores, descriptors = pool_coverage(comm, conf_round, system, pool, joinpath(dir, "params.dat"), data_points,
                                                rank=rank, nranks=nranks)
            new = Int64[]
            if rank == 0
                new = selection == "coverage" ? pool[sortperm(scores)[1:n]] : sample(pool, n, replace=false)
                if npoints_add > 0
                    candidates_desc = reduce(vcat, (descriptors[findfirst(==(i), pool)] for i in new), init=similar(data_points, 0))
                    new_points = farthest_point_extension(candidates_desc, data_points, npoints_add,
                                                          σ=get_ml_sim_params(conf), add_below=add_below)
                end
            end
        else
            new = rank == 0 ? sample(pool, n, replace=false) : Int64[]
        end
        new = MPI.bcast(new, comm, root=0)
        train = sort(vcat(train, new))

        # Initial parameters of the next round: the fitted parameters, new kernel points with weight 0
        if rank == 0 && get_ml_model(conf)
            kernel = ham.models[findfirst(m -> m isa HamiltonianKernel, ham.models)]
            next = HamiltonianKernel(vcat(kernel.params, zeros(length(new_points))), vcat(kernel.data_points, new_points),
                                     kernel.sim_params, kernel.structure_descriptors, kernel.update)
            write_params(next, conf_round, filename=joinpath(dir, "ml_params_next"))
            verbosity > 0 && !isempty(new_points) && println("  $(length(new_points)) kernel points added")
        end
        MPI.Barrier(comm)
        init_dir = dir
    end

    if rank == 0
        best = argmin(history["L_val"])
        for file in ("params.dat", basename(get_ml_filename(conf)) * ".dat")
            src = joinpath("al", "round_$best", file)
            isfile(src) && (cp(src, file * ".tmp", force=true); Base.Filesystem.rename(file * ".tmp", file))
        end
        h5open(joinpath("al", "history.h5"), "w") do file
            for (key, value) in history; file[key] = value; end
            for (round, inds) in enumerate(round_trains); file["train_config_inds/round_$round"] = inds; end
            file["val_config_inds"] = val
            file["selection"] = selection
        end
        write_al_summary(history, best, selection)
    end
    MPI.Barrier(comm)
    return prof_al
end

"""
    init_from_files!(conf, params_file, ml_file)

Initialize the TB, SOC and Ewald parameters from `params_file` and the ML model (parameters and kernel points) from
`ml_file`.
"""
function init_from_files!(conf, params_file, ml_file)
    set_value!(conf, "init_params", params_file)
    haskey(conf, "SOC") && set_value!(conf, "init_params", "SOC", params_file)
    haskey(conf, "Ewald") && set_value!(conf, "init_params", "Ewald", params_file)
    if get_ml_model(conf) && isfile(ml_file)
        set_value!(conf, "init_params", "ML", ml_file)
        get_ml_mode(conf) == "expand" && set_value!(conf, "mode", "ML", "refit")
    end
end

"""
    pool_coverage(comm, conf, system, pool, params_file, data_points; rank=0, nranks=1) -> (scores, descriptors)

Mean coverage of every pool structure by the kernel points (mean similarity of its descriptors to the most similar
point) and the descriptors of all pool structures, on rank 0 (in the order of `pool`).
"""
function pool_coverage(comm, conf, system, pool, params_file, data_points; rank=0, nranks=1)
    conf_pool = deepcopy(conf)
    set_value!(conf_pool, "init_params", params_file)
    local_pool = split_indices_into_chunks(Dict(system => pool), nranks, rank=rank)
    has_data = !isempty(local_pool)
    comm_active = active_communicator(has_data, comm, rank=rank)
    local_scores, local_descriptors = Float64[], Vector{eltype(data_points)}[]
    if has_data
        ham, _, _ = build_hamiltonian(conf_pool, comm_active, local_pool, mode="md", rank=MPI.Comm_rank(comm_active),
                                      nranks=MPI.Comm_size(comm_active), verbosity=0, write_summary=false, soc=false,
                                      ewald=false, ml_data_points=data_points)
        kernel = ham.models[findfirst(m -> m isa HamiltonianKernel, ham.models)]
        for n in eachindex(kernel.structure_descriptors)
            xs = reduce(vcat, (findnz(D)[3] for D in kernel.structure_descriptors[n]), init=eltype(data_points)[])
            push!(local_scores, mean(max_similarities(xs, data_points, kernel.sim_params)))
            push!(local_descriptors, xs)
        end
    end
    all_scores = MPI.gather(local_scores, comm, root=0)
    all_descriptors = MPI.gather(local_descriptors, comm, root=0)
    rank == 0 || return Float64[], Vector{eltype(data_points)}[]
    return reduce(vcat, all_scores), reduce(vcat, all_descriptors)
end

"""
    farthest_point_extension(candidates, points, n; σ, add_below=0.5)

Select up to `n` of the `candidates` by farthest-point sampling, starting from the existing `points`: each step takes the
candidate with the largest distance to all points selected so far, as long as its similarity `exp(-d²/2σ²)` to them is
below `add_below`.
"""
function farthest_point_extension(candidates, points, n; σ, add_below=0.5)
    selected = similar(candidates, 0)
    isempty(candidates) && return selected
    d2 = isempty(points) ? fill(Inf, length(candidates)) : tmap(x -> minimum(p -> sum(abs2, x - p), points), candidates)
    for _ in 1:n
        i = argmax(d2)
        exp(-d2[i] / (2σ^2)) < add_below || break
        push!(selected, candidates[i])
        d2 .= min.(d2, map(x -> sum(abs2, x - candidates[i]), candidates))
    end
    return selected
end

function write_al_summary(history, best, selection; filename="hamster.out")
    isfile(filename) || return nothing
    open(filename, "a") do io
        println(io)
        write_separator(io)
        println(io, "  Active Learning Summary")
        write_separator(io, char='-')
        write_summary_line(io, "selection", selection)
        println(io)
        widths = [6, 10, 8, 12, 12]
        write_table_row(io, ["round", "structures", "points", "train loss", "val loss"], widths)
        for r in eachindex(history["Ntrain"])
            write_table_row(io, [string(r), string(history["Ntrain"][r]), string(history["Npoints"][r]),
                                 format_number(history["L_train"][r], 5), format_number(history["L_val"][r], 5)], widths)
        end
        println(io)
        write_legend(io, "params.dat and ml_params.dat: round $best (lowest validation loss).")
        write_separator(io)
    end
    return nothing
end
