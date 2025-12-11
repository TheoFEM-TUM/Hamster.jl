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
- `Float64`: The minimum training loss obtained from the optimization calculation.
"""
function hyper_optimize(params, labels, prof, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), validate=get_validate(conf))::Float64
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
    prof_opt = Hamster.run_calculation(Val{:optimization}(), comm, conf, rank=rank, nranks=nranks, write_output=false)
    time = MPI.Wtime() - begin_time
    prof.timings[1, iter] = time
    Lmin = validate ? minimum(prof_opt.L_val) : minimum(prof_opt.L_train)
    
    if verbosity > 1; @printf("  Final training loss: %.6f\n", minimum(prof_opt.L_train)); end # coverage: ignore
    if verbosity > 1 && validate; @printf("  Final validation loss: %.6f\n", minimum(prof_opt.L_val)); end # coverage: ignore
    
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
    run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf))

Performs random search hyperparameter optimization by repeatedly evaluating randomly sampled parameter configurations.

# Arguments
- `::Val{:hyper_optimization}`: Dispatch tag to indicate this function handles hyperparameter optimization.
- `comm`: MPI communicator used for distributed computation.
- `conf`: Configuration object used to retrieve hyperparameter bounds, optimization settings, and verbosity.
- `rank::Int`: MPI rank (default = 0).
- `nranks::Int`: Number of MPI processes (default = 1).
- `verbosity::Int`: Controls the amount of output printed (default = retrieved from `conf`).
"""
function run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), 
            params=get_hyperopt_params(conf), lowerbounds=get_hyperopt_lowerbounds(conf), upperbounds=get_hyperopt_upperbounds(conf),
            stepsizes=get_hyperopt_stepsizes(conf), mode=get_hyperopt_mode(conf), Niter=get_hyperopt_niter(conf), log_modes = get_hyperopt_log_modes(conf))

    if verbosity == 1; set_value!(conf, "verbosity", 0); end # coverage: ignore
    
    if lowercase(mode[1]) == 'g' && Niter == 1
        param_ranges = [lower:step:upper for (lower, upper, step) in zip(lowerbounds, upperbounds, stepsizes)]
        possible_values = collect(Iterators.product(param_ranges...))
        Niter = length(possible_values)
    end
    prof = HamsterProfiler(1, conf, Niter=Niter, Nbatch=1, Nparams=length(params))
    print_start_message(prof, verbosity=verbosity)
    if lowercase(mode[1]) == 't'
        space = Dict(
            Symbol(param) => (
                log_mode == "log" ? 
                HP.LogQuantUniform(Symbol(param), log(l), log(u), δ) :
                HP.QuantUniform(Symbol(param), l, u, δ)
            )   
            for (param, l, u, δ, log_mode) in zip(params, lowerbounds, upperbounds, stepsizes, log_modes)
        )
        TreeParzen.Graph.checkspace(space)
        tpe_config = TreeParzen.Config()
        trialhist = TreeParzen.Trials.Trial[]
        for iter in 1:Niter
            trial_i = ask(space, trialhist, tpe_config)
            ps = trial_i.hyperparams
            L_local = hyper_optimize(ps, params, prof, comm, conf, rank=rank, nranks=nranks, verbosity=verbosity)
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
            L_local = hyper_optimize(param_dict, params, prof, comm, conf, rank=rank, nranks=nranks, verbosity=verbosity)
        end
    end

    if rank == 0 && verbosity > 0; println("========================================"); end

    Lmin, indmin = findmin(prof.L_train[1, :])
    # COV_EXCL_START
    if rank == 0 && verbosity > 0
        println("Final status:")
        println("  Minimal loss: $Lmin")
        for (index, param) in enumerate(params)
            println("   $param = $(prof.param_values[index, indmin])")
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