"""
    hyper_optimize(param_values, params, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf)) -> Float64

Evaluate a given set of hyperparameters by updating a configuration and running an optimization calculation.

# Arguments
- `param_values::Vector{Float64}`: Numerical values for each parameter to be optimized.
- `params::Vector{String}`: List of parameter keys. Keys can be flat (e.g., `"alpha"`) or hierarchical (e.g., `"Ga_alpha"`).
- `comm`: MPI communicator.
- `conf`: Configuration object.
- `rank::Int`: MPI rank (default = 0).
- `nranks::Int`: Total number of MPI processes (default = 1).
- `verbosity::Int`: Verbosity level (default = pulled from configuration).

# Returns
- `Float64`: The minimum of the hyperparameter optimization objective `metric` (see [`get_hyperopt_loss`](@ref)),
  or `Inf` if the optimization run of this trial crashed (the error is logged and the hyperparameter optimization continues).
"""
function hyper_optimize(params, labels, prof, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), validate=get_validate(conf), metric=get_hyperopt_metric(conf))::Float64
    if rank == 0 && verbosity > 0; println("========================================"); end # coverage: ignore

    param_values = map(labels) do label
        params[Symbol(label)]
    end
    MPI.Bcast!(param_values, comm, root=0)
    iter = findfirst(x->x==0, prof.L_train[1, :])
    for (label, param_value) in zip(labels, param_values)
        block_key = split(label, '_', limit=2)
        param_index = findfirst(l->l==label, labels)
        prof.param_values[param_index, iter] = param_value
        if length(block_key) == 1
            set_value!(conf, block_key[1], param_value)
        elseif length(block_key) == 2
            set_value!(conf, block_key[2], block_key[1], param_value)
        end
    end

    begin_time = MPI.Wtime()
    # A crashing trial (e.g., unstable hyperparameters or running out of memory) is scored with Inf instead of aborting
    # the whole hyperparameter optimization. An interrupt (Ctrl-C) still aborts.
    prof_opt, failure = try
        (Hamster.run_calculation(Val{:optimization}(), comm, conf, rank=rank, nranks=nranks, write_output=false), nothing)
    catch err
        err isa InterruptException && rethrow()
        (nothing, (err, catch_backtrace()))
    end
    time = MPI.Wtime() - begin_time
    prof.timings[1, iter] = time

    # All ranks have to agree on whether the trial failed; only the first failing rank reports the error
    Nranks = MPI.Comm_size(comm)
    first_failed = MPI.Allreduce(isnothing(failure) ? Nranks : rank, min, comm)
    if first_failed < Nranks
        if rank == first_failed; @error "Hyperopt trial $iter crashed on rank $rank and is scored with Inf." exception=failure; end
        Lmin = Inf
        GC.gc() # free the memory of the crashed run, e.g., after running out of memory
    else
        # The losses are only reduced onto rank 0, so the objective is evaluated there and shared with all ranks
        Lmin = MPI.bcast(rank == 0 ? get_hyperopt_loss(prof_opt, metric, validate=validate) : nothing, comm, root=0)
        if rank == 0 && !isfinite(Lmin); @warn "No $(validate ? "validation" : "training") $metric was recorded for this trial (check valeachiter and niter)."; end

        if verbosity > 1; @printf("  Final training loss: %.6f\n", minimum(prof_opt.L_train)); end # coverage: ignore
        if verbosity > 1 && validate; @printf("  Final validation loss: %.6f\n", minimum(prof_opt.L_val)); end # coverage: ignore
        if verbosity > 1; @printf("  Hyperopt objective (%s %s): %.6f\n", validate ? "validation" : "training", metric, Lmin); end # coverage: ignore
    end

    prof.L_train[1, iter] = Lmin
    # COV_EXCL_START
    if rank == 0 && verbosity > 0
        print_train_status(prof, iter, 1, verbosity=verbosity)
        for (key, param_value) in params
            println("   $key = $param_value")
        end
        println("Current optimum: $(minimum(prof.L_train[1, 1:iter])).")
    end
    # COV_EXCL_STOP
    return Lmin
end

"""
    get_hyperopt_loss(prof, metric="loss"; validate=false) -> Float64

Objective of the hyperparameter optimization: the minimum over all iterations of the optimization run profiled in the
`HamsterProfiler` `prof` of the metric `metric`, i.e., the training loss function (`loss`), the mean absolute error of the
eigenvalues (`mae`) or the band gap error (`bg`). It is taken from the validation set if `validate` and from the training
set otherwise. Iterations without a recorded value (after early stopping, or not validated because of `valeachiter`) are
skipped; if there are none, `Inf` is returned.
"""
function get_hyperopt_loss(prof, metric="loss"; validate=false)::Float64
    Ls = if metric == "loss"
        validate ? prof.L_val : prof.L_train
    elseif metric == "mae"
        validate ? prof.L_val_MAE : prof.L_train_MAE
    elseif metric == "bg"
        validate ? prof.L_val_BG : prof.L_train_BG
    else
        error("Unknown HyperOpt metric $metric (loss, MAE or BG).")
    end
    recorded = filter(!iszero, Ls)
    return isempty(recorded) ? Inf : minimum(recorded)
end

"""
    run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf))

Performs random search hyperparameter optimization by repeatedly evaluating randomly sampled parameter configurations.

# Arguments
- `::Val{:hyper_optimization}`: Dispatch tag to indicate this function handles hyperparameter optimization.
- `comm`: MPI communicator used for distributed computation.
- `conf`: Configuration object used to retrieve hyperparameter bounds, optimization settings, and verbosity.
- `rank::Int`: MPI rank (default = 0).
- `nranks::Int`: Number of MPI processes (default = 1).
- `verbosity::Int`: Controls the amount of output printed (default = retrieved from `conf`).
- `metric::String`: Objective that is minimized: `loss` (training loss function), `mae` or `bg` (band gap error only).
- `validate::Bool`: Whether the objective is evaluated on the validation set (otherwise the training set).
"""
function run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), 
            params=get_hyperopt_params(conf), lowerbounds=get_hyperopt_lowerbounds(conf), upperbounds=get_hyperopt_upperbounds(conf),
            stepsizes=get_hyperopt_stepsizes(conf), mode=get_hyperopt_mode(conf), Niter=get_hyperopt_niter(conf), log_modes = get_hyperopt_log_modes(conf),
            metric=get_hyperopt_metric(conf), validate=get_validate(conf))

    if verbosity == 1; set_value!(conf, "verbosity", 0); end # coverage: ignore
    
    if lowercase(mode[1]) == 'g' && Niter == 1
        param_ranges = [lower:step:upper for (lower, upper, step) in zip(lowerbounds, upperbounds, stepsizes)]
        possible_values = collect(Iterators.product(param_ranges...))
        Niter = length(possible_values)
    end
    prof = HamsterProfiler(1, conf, Niter=Niter, Nbatch=1, Nparams=length(params))
    print_start_message(prof, verbosity=verbosity)

    if lowercase(mode[1]) == 't'
        space = Dict(Symbol(param)=>HP.QuantUniform(Symbol(param), l, u, δ) for (param, l, u, δ) in zip(params, lowerbounds, upperbounds, stepsizes))
        TreeParzen.Graph.checkspace(space)
        tpe_config = TreeParzen.Config()
        trialhist = TreeParzen.Trials.Trial[]
        for iter in 1:Niter
            trial_i = ask(space, trialhist, tpe_config)
            ps_raw = trial_i.hyperparams

            ps = Dict(
                k => (
                    log_modes[findfirst(x -> x == String(k), params)] == "log" ?
                    10.0^v :
                    v
                )
                for (k, v) in ps_raw
            )
            L_local = hyper_optimize(ps, params, prof, comm, conf, rank=rank, nranks=nranks, verbosity=verbosity, validate=validate, metric=metric)
            tell!(trialhist, trial_i, L_local)
        end
    else
        for iter in 1:Niter
            if lowercase(mode[1]) == 'r'
                param_values = [rand(lower:step:upper) for (lower, upper, step) in zip(lowerbounds, upperbounds, stepsizes)]
            elseif lowercase(mode[1]) == 'g'
                param_ranges = [lower:step:upper for (lower, upper, step) in zip(lowerbounds, upperbounds, stepsizes)]
                possible_values = collect(Iterators.product(param_ranges...))
                param_values = [possible_values[iter]...]
            end
            param_dict = Dict(Symbol(param)=>value for (param, value) in zip(params, param_values))
            L_local = hyper_optimize(param_dict, params, prof, comm, conf, rank=rank, nranks=nranks, verbosity=verbosity, validate=validate, metric=metric)
        end
    end

    if rank == 0 && verbosity > 0; println("========================================"); end

    Lmin, indmin = findmin(prof.L_train[1, :])
    Nfailed = count(isinf, prof.L_train[1, :])
    if rank == 0 && Nfailed > 0; @warn "$Nfailed of $Niter hyperopt trials failed and were scored with Inf."; end
    # COV_EXCL_START
    if rank == 0 && verbosity > 0
        println("Final status:")
        if isinf(Lmin)
            println("  All trials failed.")
        else
            println("  Minimal $(validate ? "validation" : "training") $metric: $Lmin")
            for (index, param) in enumerate(params)
                println("   $param = $(prof.param_values[index, indmin])")
            end
        end
        println("========================================")
    end
    # COV_EXCL_STOP

    if rank == 0
        h5open("hamster_out.h5", "w") do file
            for (index, param) in enumerate(params)
                file[param] = prof.param_values[index, :]
            end
            file["params"] = params
        end
    end

    return prof
end