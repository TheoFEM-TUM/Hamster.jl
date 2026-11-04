"""
    hyper_optimize(params, labels, tags, prof, L_fit, iter, comm, conf; config_inds=nothing, rank=0, nranks=1,
                   verbosity=get_verbosity(conf), validate=get_validate(conf)) -> Float64

Evaluate the candidate `iter` of a hyperparameter optimization: set the hyperparameters in `conf`, run a fit and store
the result in `prof` (on rank 0) and `L_fit`.

# Arguments
- `params`: values of the hyperparameters by label (`Dict{Symbol, Float64}`).
- `labels::Vector{String}`: labels of the hyperparameters, as in the `params` tag of the HyperOpt block.
- `tags`: input tag (`(name, block)`) of each label, see [`Hamster.hyperopt_tag`](@ref).
- `prof`: profiler of the hyperparameter optimization: `L_train[1, iter]` gets the score of the candidate,
  `L_val[iter]` its lowest validation loss, `param_values[:, iter]` its hyperparameters and `timings[1, iter]` the time
  of the fit.
- `L_fit`: `L_fit[iter]` gets the lowest training loss of the fit.
- `config_inds`: training and validation configuration indices of all candidates (from
  `get_config_inds_for_systems`); `nothing` samples them in the fit.

# Returns
- `Float64`: the score of the candidate (on all ranks): the lowest validation loss of the fit if `validate = true`,
  else the lowest training loss.
"""
function hyper_optimize(params, labels, tags, prof, L_fit, iter, comm, conf; config_inds=nothing, rank=0, nranks=1,
                        verbosity=get_verbosity(conf), validate=get_validate(conf), valeachiter=get_valeachiter(conf))::Float64
    if rank == 0 && verbosity > 0; println("========================================"); end # coverage: ignore

    param_values = Float64[params[Symbol(label)] for label in labels]
    MPI.Bcast!(param_values, comm, root=0)
    for (index, ((name, block), param_value)) in enumerate(zip(tags, param_values))
        prof.param_values[index, iter] = param_value
        if block == "Options"
            set_value!(conf, name, param_value)
        else
            set_value!(conf, name, block, param_value)
        end
    end

    begin_time = MPI.Wtime()
    prof_opt = Hamster.run_calculation(Val{:optimization}(), comm, conf, rank=rank, nranks=nranks, write_output=false,
                                       config_inds=config_inds)
    prof.timings[1, iter] = MPI.Wtime() - begin_time

    L_train_min = minimum(prof_opt.L_train)
    validated = prof_opt.L_val[1:valeachiter*(length(prof_opt.L_val) ÷ valeachiter)]
    L_val_min = validate ? (isempty(validated) ? Inf : minimum(validated)) : 0.
    score = MPI.bcast(validate ? L_val_min : L_train_min, comm, root=0)
    L_fit[iter] = L_train_min
    prof.L_val[iter] = L_val_min
    prof.L_train[1, iter] = score

    # COV_EXCL_START
    if rank == 0 && verbosity > 0
        if verbosity > 1
            @printf("  Final training loss: %.6f\n", L_train_min)
            validate && @printf("  Final validation loss: %.6f\n", L_val_min)
        end
        println("Candidate $iter / $(size(prof.L_train, 2)) | $(validate ? "Validation" : "Training") loss: $score | Time: $(prof.timings[1, iter]) s")
        for (label, param_value) in zip(labels, param_values)
            println("   $label = $param_value")
        end
        println("Current optimum: $(minimum(prof.L_train[1, 1:iter])).")
        flush(stdout)       # stdout redirected to a file (jobs) is buffered otherwise
    end
    # COV_EXCL_STOP
    return score
end

"""
    hyperopt_tag(label) -> (name, block)

Input tag of the hyperparameter `label` from the `params` tag of the HyperOpt block. A label is the name of a tag of the
Options block (`rcut`) or of a tag that exists in only one other block (`lr`), `Block_tag` for a tag of a block
(`ML_rcut`, `Optimizer_val_ratio`), or `Element_tag` for a tag of an atom block (`Ga_alpha`). `block` is `"Options"`,
the block name or the element. Unknown and ambiguous labels are an error.
"""
function hyperopt_tag(label)
    known(name, block) = !isempty(find_tags(name, block=block)) || is_indexed_weight(name, block) || is_site_charge(name, block)
    blocks = [b[1] for b in CONFIG_BLOCKS if b[1] ∉ ("atom", "HyperOpt")]
    candidates = Tuple{String, String}[]
    known(label, "Options") && push!(candidates, (label, "Options"))
    parts = split(label, '_', limit=2)
    if length(parts) == 2
        prefix, name = String.(parts)
        for block in blocks
            lowercase(block) == lowercase(prefix) && known(name, block) && push!(candidates, (name, block))
        end
        is_atom_block(prefix) && known(name, "atom") && push!(candidates, (name, prefix))
    end
    tag_blocks = unique(tag.block for tag in find_tags(label) if tag.block != "HyperOpt")
    is_indexed_weight(label, "Optimizer") && push!(tag_blocks, "Optimizer")
    if isempty(candidates) && length(tag_blocks) == 1 && tag_blocks[1] != "atom"
        push!(candidates, (label, tag_blocks[1]))
    end
    length(candidates) == 1 && return candidates[1]

    where_ = "in the params tag of the HyperOpt block"
    if length(candidates) > 1
        error("Hyperparameter \"$label\" $where_ is ambiguous; it can be " *
              join([block == "Options" ? name : "$(block)_$name" for (name, block) in candidates], " or ") * ".")
    elseif tag_blocks == ["atom"]
        error("Hyperparameter \"$label\" $where_ is set per species; give it as <element>_$label (e.g. Ga_$label).")
    elseif length(tag_blocks) > 1
        error("Hyperparameter \"$label\" $where_ exists in several blocks; give it as " *
              join([block == "atom" ? "<element>_$label" : "$(block)_$label" for block in tag_blocks], " or ") * ".")
    else
        error("Unknown hyperparameter \"$label\" $where_; give it as <tag> (Options block), <Block>_<tag> " *
              "(e.g. ML_rcut) or <element>_<tag> (e.g. Ga_alpha).")
    end
end

"""
    check_hyperopt_settings(params, lowerbounds, upperbounds, stepsizes, mode)

Errors for settings of the HyperOpt block that do not fit together: lists of different lengths, a lower bound above the
upper bound, step sizes that are not positive and an unknown `mode`.
"""
function check_hyperopt_settings(params, lowerbounds, upperbounds, stepsizes, mode)
    isempty(params) && error("The params tag of the HyperOpt block is empty; set the hyperparameters to optimize.")
    for (tag, values) in (("lowerbounds", lowerbounds), ("upperbounds", upperbounds), ("stepsizes", stepsizes))
        length(values) == length(params) ||
            error("The $tag tag of the HyperOpt block has $(length(values)) values for $(length(params)) hyperparameters (params).")
    end
    for (param, lower, upper, step) in zip(params, lowerbounds, upperbounds, stepsizes)
        lower ≤ upper || error("Hyperparameter $param: the lower bound $lower is larger than the upper bound $upper (HyperOpt block).")
        step > 0 || error("Hyperparameter $param: the step size must be positive, got $step (stepsizes in the HyperOpt block).")
    end
    lowercase(mode) ∈ ("random", "grid", "tpe") ||
        error("Unknown mode \"$mode\" in the HyperOpt block; use random, grid or tpe.")
    return nothing
end

"""
    split_depends_on(tags) -> Bool

Whether one of the hyperparameter tags changes which configurations are sampled for training and validation (tags of
the Supercell block, `val_ratio`, `validate`, `train_mode`, `val_mode`). Then the configurations are sampled again for
every candidate.
"""
split_depends_on(tags) = any(((name, block),) -> block == "Supercell" ||
                             (block == "Optimizer" && lowercase(name) ∈ ("val_ratio", "validate", "train_mode", "val_mode")), tags)

"""
    write_hyperopt_results(prof, L_fit, labels, Nevaluated; filename="hamster_out.h5")

Write the results of a hyperparameter optimization to `filename`: the profiler (`L_train`: score of each candidate,
`L_val`, `timings`, `param_values`), `L_fit` (lowest training loss of each candidate), a dataset with the values of each
hyperparameter, the list of hyperparameters (`params`) and the number of evaluated candidates (attribute `evaluated`).
Candidates that were not evaluated yet have zeros.
"""
function write_hyperopt_results(prof, L_fit, labels, Nevaluated; filename="hamster_out.h5")
    save(prof, filename=filename)
    h5open(filename, "cw") do file
        for (index, label) in enumerate(labels)
            replace_dataset!(file, label, prof.param_values[index, :])
        end
        replace_dataset!(file, "params", labels)
        replace_dataset!(file, "L_fit", L_fit)
        replace_attribute!(file, "evaluated", Nevaluated)
    end
end

"""
    run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf))

Perform a hyperparameter optimization by evaluating the model over many parameter configurations and identifying the one that yields the lowest loss.

This routine drives the full hyperparameter-search workflow, including parameter sampling, model evaluation, bookkeeping, and reporting of results. 
It supports random search, grid search, and Tree-Parzen Estimator (TPE; Bayesian optimization).

# Workflow

1. **Check the settings and determine the number of candidates.**  
   The hyperparameter labels are resolved to input tags (see [`Hamster.hyperopt_tag`](@ref)) and the lists of the
   `HyperOpt` block are checked. Random search and TPE evaluate `niter` candidates, grid search evaluates all points
   of the grid (`niter` is not used).

2. **Sample the training and validation configurations.**  
   The configurations are sampled once and used for all candidates, so that all candidates are compared on the same
   data. Only if a hyperparameter changes the sampling (a tag of the `Supercell` block, `val_ratio`, `validate`,
   `train_mode` or `val_mode`), they are sampled again for every candidate.

3. **Construct the search space.**  
   - For **TPE**, a probabilistic search space is built using quantized uniform distributions.
   - For **random** and **grid** search, discrete candidate values are generated directly from `lowerbounds`, `upperbounds` and `stepsizes`.

4. **Iterative evaluation.**  
   For each candidate:
   - A set of hyperparameters is sampled from the search space.  
   - A fit is run with these parameters using `hyper_optimize`, which returns the score of the candidate: the lowest
     validation loss of the fit if the fit is validated, else the lowest training loss.  
   - For TPE, the sampled point and its score update the optimization history to refine future proposals.
   - The results so far are written to `hamster_out.h5`, so they are kept if the run stops early.

5. **Select the best configuration.**  
   After all candidates, the routine identifies the parameter set with the lowest score and prints a summary (if `verbosity > 0`).

# Required Inputs

- **Training data set** (see `Optimizer`).  
- **Validation data set** (optional; see `Optimizer`).

# Settings (from the `HyperOpt` block)

- `params` – Names of the hyperparameters to optimize: `tag` for a tag of the `Options` block (or a tag that exists
  in only one other block), `Block_tag` for a tag of another block (e.g. `ML_rcut`) and `Element_tag` for a tag of an
  atom block (e.g. `Ga_alpha`).  
- `lowerbounds`, `upperbounds` – Numerical bounds for each parameter.  
- `stepsizes` – Step size used for random/grid sampling and quantization.  
- `niter` – Number of candidates for random search and TPE.  
- `mode` – Sampling strategy: `random`, `grid`, or `tpe`.

# Output Files

- `hamster.out` – Standard Hamster output.  
- `hamster_out.h5` – HDF5 file, written after every candidate, containing:
  - `L_train`: the score of each candidate (the lowest validation loss if the fits are validated, else the lowest
    training loss),
  - `L_val`: the lowest validation loss of each candidate (zero without validation),
  - `L_fit`: the lowest training loss of each candidate,
  - `timings`: the time of each fit,
  - one dataset per hyperparameter with its value for each candidate, and `params`, the list of hyperparameters,
  - `train_config_inds` and `val_config_inds`: the configurations of the fits (if they are the same for all candidates),
  - the attribute `evaluated`: the number of evaluated candidates (the entries of later candidates are zero).
"""
function run_calculation(::Val{:hyper_optimization}, comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), 
            params=get_hyperopt_params(conf), lowerbounds=get_hyperopt_lowerbounds(conf), upperbounds=get_hyperopt_upperbounds(conf),
            stepsizes=get_hyperopt_stepsizes(conf), mode=get_hyperopt_mode(conf), Niter=get_hyperopt_niter(conf))

    check_hyperopt_settings(params, lowerbounds, upperbounds, stepsizes, mode)
    tags = hyperopt_tag.(params)
    mode = lowercase(mode)
    if verbosity == 1; set_value!(conf, "verbosity", 0); end # coverage: ignore

    param_ranges = [lower:step:upper for (lower, upper, step) in zip(lowerbounds, upperbounds, stepsizes)]
    if mode == "grid"
        grid = vec(collect(Iterators.product(param_ranges...)))
        Niter = length(grid)
    end
    prof = HamsterProfiler(1, conf, Niter=Niter, Nbatch=1, Nparams=length(params))
    L_fit = zeros(Niter)
    print_start_message(prof, verbosity=verbosity)

    config_inds = split_depends_on(tags) ? nothing :
                  get_config_inds_for_systems(get_systems(conf), comm, conf, rank=rank, write_output=true, optimize=true)

    evaluate(param_dict, iter) = hyper_optimize(param_dict, params, tags, prof, L_fit, iter, comm, conf,
                                                config_inds=config_inds, rank=rank, nranks=nranks, verbosity=verbosity)
    if mode == "tpe"
        space = Dict(Symbol(param)=>HP.QuantUniform(Symbol(param), l, u, δ) for (param, l, u, δ) in zip(params, lowerbounds, upperbounds, stepsizes))
        TreeParzen.Graph.checkspace(space)
        tpe_config = TreeParzen.Config()
        trialhist = TreeParzen.Trials.Trial[]
    end
    for iter in 1:Niter
        if mode == "tpe"
            trial_i = ask(space, trialhist, tpe_config)
            score = evaluate(trial_i.hyperparams, iter)
            tell!(trialhist, trial_i, score)
        else
            param_values = mode == "random" ? rand.(param_ranges) : collect(grid[iter])
            evaluate(Dict(Symbol(param)=>value for (param, value) in zip(params, param_values)), iter)
        end
        if rank == 0
            write_hyperopt_results(prof, L_fit, params, iter)
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
        flush(stdout)
    end
    # COV_EXCL_STOP

    return prof
end
