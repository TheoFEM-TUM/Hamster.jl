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
function hyper_optimize(param_values, params, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), validate=get_validate(conf))::Float64
    for (index, param) in enumerate(params)
        block_key = split_line(param, char="_")
        if length(block_key) == 1
            set_value!(conf, block_key[1], param_values[index])
        elseif length(block_key) == 2
            set_value!(conf, block_key[2], block_key[1], param_values[index])
        end
    end
    prof = Hamster.run_calculation(Val{:optimization}(), comm, conf, rank=rank, nranks=nranks)
    Lmin = validate ? minimum(prof.L_val) : minimum(prof.L_train)
    if verbosity > 1; @printf("  Final training loss: %.6f\n", minimum(prof.L_train)); end # coverage: ignore
    if verbosity > 1 && validate; @printf("  Final validation loss: %.6f\n", minimum(prof.L_val)); end # coverage: ignore
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
function run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf))
    params = get_hyperopt_params(conf)
    lowerbounds = get_hyperopt_lowerbounds(conf)
    upperbounds = get_hyperopt_upperbounds(conf)
    stepsizes = get_hyperopt_stepsizes(conf, length(params))
    Niter = get_hyperopt_niter(conf)

    all_params = zeros(length(params), Niter)
    if verbosity == 1; set_value!(conf, "verbosity", 0); end # coverage: ignore
    prof = HamsterProfiler(1, conf, Niter=Niter, Nbatch=1)

    for iter in 1:Niter
        if rank == 0 && verbosity > 0; println("========================================"); end # coverage: ignore
        param_values = [rand(lower:step:upper) for (lower, upper, step) in zip(lowerbounds, upperbounds, stepsizes)]
        MPI.Bcast!(param_values, comm, root=0)

        all_params[:, iter] = param_values
        begin_time = MPI.Wtime()
        L_local = hyper_optimize(param_values, params, comm, conf, rank=rank, nranks=nranks, verbosity=verbosity)
        time = MPI.Wtime() - begin_time
        prof.timings[1, iter] = time
        prof.L_train[1, iter] = L_local

        # coverage: ignore start
        if rank == 0 && verbosity > 0
            print_train_status(prof, iter, 1, verbosity=verbosity)
            for (index, param) in enumerate(params)
                println("   $param = $(all_params[index, iter])")
            end
            println("Current optimum: $(minimum(prof.L_train[1, 1:iter])).")
        end
        # coverage: ignore end
    end
    if rank == 0 && verbosity > 0; println("========================================"); end

    Lmin, indmin = findmin(prof.L_train[1, :])
    # coverage: ignore start
    if rank == 0 && verbosity > 0
        println("Final status:")
        println("  Minimal loss: $Lmin")
        for (index, param) in enumerate(params)
            println("   $param = $(all_params[index, indmin])")
        end
        println("========================================")
    end
    # coverage: ignore end

    h5open("hyperopt_out.h5", "w") do file
        file["L_train"] = prof.L_train[1, :]
        file["param_values"] = all_params
        file["params"] = params
    end
    return prof
end