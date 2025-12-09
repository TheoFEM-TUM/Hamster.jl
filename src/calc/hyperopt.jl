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

Perform a hyperparameter optimization by evaluating the model over many parameter configurations and identifying the one that yields the lowest loss.

This routine drives the full hyperparameter-search workflow, including parameter sampling, model evaluation, bookkeeping, and reporting of results. 
It supports random search, grid search, and Tree-Parzen Estimator (TPE; Bayesian optimization).

# Workflow

1. **Determine search mode and iterations.**  
   The algorithm selects the sampling strategy (`random`, `grid`, or `tpe`) and computes the total number of evaluations. For grid search, if `niter = 1`, the full Cartesian product of parameter ranges is used.

2. **Construct the search space.**  
   Parameter names, bounds, and step sizes from the `HyperOpt` block define the domain of the optimization.  
   - For **TPE**, a probabilistic search space is built using quantized uniform distributions.
   - For **random** and **grid** search, discrete candidate values are generated directly from `lowerbounds`, `upperbounds` and `stepsizes`.

3. **Iterative evaluation.**  
   For each iteration:
   - A set of hyperparameters is sampled from the search space.  
   - The model is executed with these parameters using `hyper_optimize`, which returns the training loss.  
   - The profiler records the parameter values and corresponding losses.  
   - For TPE, the sampled point and observed loss update the optimization history to refine future proposals.

4. **Select the best configuration.**  
   After all iterations, the routine identifies the parameter set that achieved the minimal training loss and prints a summary (if `verbosity > 0`).

5. **Write output.**  
   On rank 0, the sampled parameter values and metadata are written to `hamster_out.h5`, and standard output is written to `hamster.out`.

# Required Inputs

- **Training data set** (see `Optimizer`).  
- **Validation data set** (optional; see `Optimizer`).

# Settings (from the `HyperOpt` block)

- `params` – Names of hyperparameters to optimize.  
  *The substring before the first `_` is interpreted as the corresponding block name.*  
- `lowerbounds`, `upperbounds` – Numerical bounds for each parameter.  
- `stepsizes` – Step size used for random/grid sampling and quantization.  
- `niter` – Number of iterations (ignored for full grid expansion if `niter = 1`).  
- `mode` – Sampling strategy: `random`, `grid`, or `tpe`.

# Output Files

- `hamster.out` – Standard Hamster output.  
- `hamster_out.h5` – HDF5 file containing:
  - parameter values for all evaluated configurations,
  - associated loss values,
  - the list of optimized parameters.
"""
function run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), 
            params=get_hyperopt_params(conf), lowerbounds=get_hyperopt_lowerbounds(conf), upperbounds=get_hyperopt_upperbounds(conf),
            stepsizes=get_hyperopt_stepsizes(conf), mode=get_hyperopt_mode(conf), Niter=get_hyperopt_niter(conf))

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