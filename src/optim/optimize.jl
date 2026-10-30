"""
    optimize_model!(ham_train, ham_val, optim, dl, prof, conf=get_empty_config(); verbosity=get_verbosity(conf), Nbatch=get_nbatch(conf), validate=get_validate(conf), rank=0, nranks=1)

Optimizes the model by performing training and optional validation steps.

# Arguments
- `ham_train`: The Hamiltonian model used for training.
- `ham_val`: The Hamiltonian model used for validation (optional).
- `optim`: An optimization configuration, including the optimizer and its settings.
- `dl`: A data loader object containing the training data.
- `prof`: A profiler object used to store training and validation information.
- `comm`: The MPI communicator.
- `conf`: A `Config` instance.
- `verbosity`: The level of verbosity for logging.
- `Nbatch`: The number of batches per training iteration.
- `validate`: A flag indicating whether to perform validation during training.
- `shuffle_batches`: A flag indicating whether the structures are shuffled before they are split into batches in every iteration.

# Workflow
1. Print the start message.
2. For each training iteration, split the training data into batches and perform training steps.
3. Optionally validate the model after each training iteration.
4. Print the final status once training is complete.

# Returns
- Updates the HamsterProfiler `prof` and the model parameters in `ham_train` and `ham_val`.
"""
function optimize_model!(ham_train, ham_val, optim, dl, prof, comm, conf=get_empty_config(); verbosity=get_verbosity(conf), Nbatch=get_nbatch(conf), validate=get_validate(conf), valeachiter=get_valeachiter(conf), shuffle_batches=get_shuffle_batches(conf), rank=0, nranks=1,
                         patience=get_early_stopping(conf), es_metric=get_early_stopping_metric(conf), restore_best=get_restore_best(conf))
    print_start_message(prof; verbosity=verbosity)
    early_stopping = patience > 0 && validate
    if patience > 0 && !validate && rank == 0; @warn "early_stopping requires validate = true and is switched off."; end
    # Best validation metric, its iteration and the parameters and energy offsets at that point
    best_value, best_iter = Inf, 0
    best_params = [copy(get_params(model)) for model in ham_train.models]
    best_offsets = [l.offset for l in optim.losses]
    last_iter = optim.Niter
    for iter in 1:optim.Niter
        iter_begin = MPI.Wtime()
        strc_indices = shuffle_batches ? shuffle(1:ham_train.Nstrc) : (1:ham_train.Nstrc)
        for (batch_id, indices) in enumerate(chunks(strc_indices, n=Nbatch))
            train_step!(ham_train, indices, optim, dl.train_data, prof, iter, batch_id, comm, conf, rank=rank, nranks=nranks)
            print_train_status(prof, iter, batch_id, verbosity=verbosity)
            #if rank == 0;@info "NEXT";end
        end
        if validate && mod(iter, valeachiter) == 0
            print_val_start(prof, iter, verbosity=verbosity)
            copy_params!(ham_val, ham_train)
            val_step!(ham_val, optim.val_losses, dl.val_data, prof, iter, comm, rank=rank, nranks=nranks, valeachiter=valeachiter, conf = conf)
            print_val_status(prof, iter, verbosity=verbosity)
        end
        stop = false
        if early_stopping && mod(iter, valeachiter) == 0
            value = MPI.bcast(rank == 0 ? get_val_metric(prof, iter, es_metric) : nothing, comm, root=0)
            if value < best_value
                best_value, best_iter = value, iter
                best_params = [copy(get_params(model)) for model in ham_train.models]
                best_offsets = [l.offset for l in optim.losses]
            elseif iter - best_iter ≥ patience
                stop = true
            end
        end
        MPI.Barrier(comm)
        iter_time = MPI.Wtime() - iter_begin
        if verbosity > 1 && rank == 0; println("Iteration time: $iter_time s"); end
        if stop
            last_iter = iter
            if verbosity > 0 && rank == 0; println("Early stopping at iteration $iter: no improvement of the validation $es_metric since iteration $best_iter."); end
            break
        end
    end
    final_iter = last_iter
    if early_stopping && restore_best && 0 < best_iter < last_iter
        for (model, params) in zip(ham_train.models, best_params); set_params!(model, params); end
        for (l, offset) in zip(optim.losses, best_offsets); l.offset = offset; end
        final_iter = best_iter
        if verbosity > 0 && rank == 0; println("Restored the parameters of iteration $best_iter (validation $es_metric: $(round(best_value, digits=6)))."); end
    end
    if get_offset(conf) && get_onsite_shift(conf) == "end"
        shift_onsite_params!(ham_train, optim.losses, comm, rank=rank, verbosity=verbosity)
    end
    print_final_status(prof; verbosity=verbosity, iter=final_iter)
end

"""
    get_val_metric(prof, iter, metric)

Validation metric `metric` (`loss`, `mae` or `bg`) of iteration `iter` stored in the `HamsterProfiler` `prof`.
"""
function get_val_metric(prof, iter, metric)
    metric == "loss" && return prof.L_val[iter]
    metric == "mae" && return prof.L_val_MAE[iter]
    metric == "bg" && return prof.L_val_BG[iter]
    error("Unknown early_stopping_metric $metric (loss, MAE or BG).")
end

"""
    shift_onsite_params!(ham, losses, comm; rank=0, verbosity=1)

Subtracts the mean energy offset across systems from the diagonal onsite parameters of the `TBModel` in `ham`. The
offsets (TB minus reference) are averaged per system first, so that every system counts once. As the diagonal onsites
enter H as an identity block, this shifts the TB spectra rigidly onto the reference energies on average. The offsets
stored in `losses` are shifted accordingly. Nothing is done if the TB parameters are not updated.
"""
function shift_onsite_params!(ham, losses, comm; rank=0, verbosity=1)
    any(model -> model isa TBModel && any(model.update), ham.models) || return
    all_systems_offsets = MPI.gather([(l.system, l.offset) for l in losses], comm, root=0)
    Δ = Ref(0.0)
    if rank == 0
        sums = Dict{String, Float64}()
        counts = Dict{String, Int}()
        for (system, offset) in reduce(vcat, all_systems_offsets)
            sums[system] = get(sums, system, 0.0) + offset
            counts[system] = get(counts, system, 0) + 1
        end
        Δ[] = mean(sums[system] / counts[system] for system in keys(sums))
    end
    Δ = MPI.Bcast!(Δ, 0, comm)[]

    for model in ham.models
        model isa TBModel || continue
        for (v, label) in enumerate(model.param_labels)
            if label.nnlabel == 0 && label.overlap_label[3] == 0
                model.params[v] -= Δ
            end
        end
    end
    for l in losses; l.offset -= Δ; end
    if rank == 0 && verbosity > 1; println(" Onsite shift: $(-Δ) eV"); end
end

"""
    load_simmats!(ham::EffectiveHamiltonian, indices)
    unload_simmats!(ham::EffectiveHamiltonian, indices)

Loads / unloads the `SimMat`s of the structures `indices` for every `HamiltonianKernel` in `ham.models`.
"""
function load_simmats!(ham::EffectiveHamiltonian, indices)
    for model in ham.models
        model isa HamiltonianKernel && load_simmats!(model, indices)
    end
    return ham
end

function unload_simmats!(ham::EffectiveHamiltonian, indices)
    for model in ham.models
        model isa HamiltonianKernel && unload_simmats!(model, indices)
    end
    return ham
end

"""
    train_step!(ham_train, indices, optim, train_data)

Performs a single training step on a Hamiltonian model by computing gradients and updating model parameters.

# Arguments
- `ham_train`: The Hamiltonian model being trained.
- `indices`: The indices of the structures to be evaluated.
- `optim`: A `GDOptimizer` instance.
- `train_data`: The training data.
- `prof`: A `HamsterProfiler` instance.
- `iter`: The iteration index.
- `batch_id`: The batch index.
- `comm`: The MPI communicator.
- `conf`: A `Config` instance.
- `rank`: The active MPI rank.
- `nranks`: The total number of MPI ranks.

# Side Effects
- Updates the model parameters in-place within `ham_train`.
- Writes timing information and training loss to `prof`.
"""
function train_step!(ham_train, indices, optim, train_data, prof, iter, batch_id, comm, conf=get_empty_config(); 
    rank=0, 
    nranks=1,
    lr=get_lr(conf),
    lr_min=get_lr_min(conf),
    verbosity=get_verbosity(conf),
    warmup_ratio=get_warmup_ratio(conf),
    lr_warmup=get_lr_warmup(conf),
    offset_step = get_offset_step(conf),
    ls_weight = get_ls_weight(conf),
    nbatch = get_nbatch(conf),
    onsite_shift = get_onsite_shift(conf)
    )

    if rank == 0 && offset_step == iter
        println("-------------Iter $iter : offset freeze------------")
    end

    warmup = max(1.0, warmup_ratio * optim.Niter)
    if iter < warmup
        lr_start = lr_min * 0.01
        x = iter / warmup
        optim.adam.eta = lr_start * (lr / lr_start)^x
        #optim.adam.eta = lr * iter / warmup
    else
        progress = (iter - warmup) / (optim.Niter - warmup)
        optim.adam.eta =
            lr_min + 0.5 * (lr - lr_min) * (1 + cos(π * progress))
    end
    lr_log = optim.adam.eta

    # The device path keeps the linear maps of all structures, so the SimMats are only needed for real-space data
    load_sms = nbatch > 1 && !(uses_device_path(ham_train) && train_data isa AbstractVector{<:EigData})
    if load_sms; load_simmats!(ham_train, indices); end

    forward_times = Float64[]
    backward_times = Float64[]
    Ls_train = Float64[]
    Ls_train_MAE = Float64[]
    Ls_train_BG = Float64[]

    N = length(indices)
    caches = Vector{Any}(undef, N)
    L_trains_weights = Vector{Float64}(undef, N)
    pc_weights = Float64[]
    systems_offsets = Tuple{String, Float64}[]
    for (i, index) in enumerate(indices)
        f_time = @elapsed cache, offset = forward(ham_train, index, train_data[index], optim.losses[index])
        caches[i] = cache
        push!(forward_times, f_time)
        push!(systems_offsets, (optim.losses[index].system, offset))
    end
    if offset_step >= iter
        elapsed = @elapsed begin
            all_systems_offsets = MPI.gather(systems_offsets, comm, root = 0)

            # Flatten on rank 0
            if rank == 0
                all_systems_offsets = reduce(vcat, all_systems_offsets)
            else
                all_systems_offsets = nothing
            end

            # Broadcast to all ranks
            all_systems_offsets = MPI.bcast(all_systems_offsets, comm, root = 0)

            # Every rank computes the averages
            sums = Dict{String, Float64}()
            counts = Dict{String, Int}()

            for (system, offset) in all_systems_offsets
                sums[system] = get(sums, system, 0.0) + offset
                counts[system] = get(counts, system, 0) + 1
            end

            systems_offsets_averages = Dict(
                system => sums[system] / counts[system]
                for system in keys(sums)
            )
        end
    else 
        systems_offsets_averages = nothing
        elapsed = 0
    end


    for (i, index) in enumerate(indices)
        if offset_step >= iter
            optim.losses[index].offset = systems_offsets_averages[optim.losses[index].system]
        end
        f_time = @elapsed L_train, L_train_MAE, L_train_BG = forward(caches[i][1], optim.losses[index], train_data[index])
        forward_times[i] += (f_time + elapsed)
        push!(Ls_train, L_train)
        push!(Ls_train_MAE, L_train_MAE * optim.losses[index].pc_weight)
        push!(Ls_train_BG, L_train_BG * optim.losses[index].pc_weight)
        push!(pc_weights, optim.losses[index].pc_weight)
    end
    pc_weights_tot = MPI.Allreduce(sum(pc_weights), +, comm)
    L_train_sum = ls_weight ? MPI.Allreduce(sum(Ls_train .* pc_weights), +, comm) : MPI.Allreduce(sum(ones(size(Ls_train)) .* pc_weights), +, comm)
    
    L_trains_weights = ls_weight ? Ls_train ./ (L_train_sum / pc_weights_tot) .* pc_weights : ones(size(Ls_train)) ./ (L_train_sum / pc_weights_tot) .* pc_weights
    Ls_train = L_trains_weights .* Ls_train

    dL_dHr = map(enumerate(indices)) do (i, index)
        b_time = @elapsed dL_dHr_index = backward(L_trains_weights[i], ham_train, index, optim.losses[index], train_data[index], caches[i], conf)
        push!(backward_times, b_time)
        return dL_dHr_index
    end
    all_systems = MPI.gather(ham_train.systems[indices], comm, root=0)
    all_losses = MPI.gather(Ls_train, comm, root=0)
    all_losses_MAE = MPI.gather(Ls_train_MAE, comm, root=0)
    all_losses_BG = MPI.gather(Ls_train_BG, comm, root=0)
    if rank == 0
        all_systems = vcat(all_systems...)
        all_losses = vcat(all_losses...)
        all_losses_MAE = vcat(all_losses_MAE...)
        all_losses_BG = vcat(all_losses_BG...)

        for system in unique(all_systems)
            if !haskey(prof.L_train_system, system)
                prof.L_train_system[system] = zeros(size(prof.L_train))
                prof.L_train_system_MAE[system] = zeros(size(prof.L_train_MAE))
                prof.L_train_system_BG[system] = zeros(size(prof.L_train_BG))
            end
            idxs = findall(s -> s == system, all_systems)
            loss_system = all_losses[idxs]
            loss_system_MAE = all_losses_MAE[idxs]
            loss_system_BG = all_losses_BG[idxs]
            prof.L_train_system[system][batch_id, iter] = mean(loss_system)
            prof.L_train_system_MAE[system][batch_id, iter] = mean(loss_system_MAE)
            prof.L_train_system_BG[system][batch_id, iter] = mean(loss_system_BG)
        end
    end
    update_begin = MPI.Wtime()
    for (model_index, model) in enumerate(ham_train.models)
        model_grad_local = get_model_gradient(model, indices, optim.reg, model_gradient_input(dL_dHr, model_index); soc=ham_train.soc)
        model_grad = MPI.Reduce(model_grad_local, +, comm, root=0)
        if rank == 0; update!(model, optim.adam, model_grad ./ pc_weights_tot); end
        params = get_params(model)
        #if rank == 0; @info "NZ params $(count(iszero, params))" ; end
        #if rank == 0; @info "NZ grad params $(count(iszero, model_grad))" ; end
        MPI.Bcast!(params, comm, root=0)
        set_params!(model, params)
    end
    if get_offset(conf) && onsite_shift == "step"
        shift_onsite_params!(ham_train, optim.losses, comm, rank=rank, verbosity=verbosity)
    end
    #optim.adam.eta = lr_min + 0.5 * (lr - lr_min) * (1 + cos(π * iter / optim.Niter))
    update_time_local = MPI.Wtime() - update_begin

    #L_train = MPI.Reduce(sum(Ls_train.^2), +, comm, root=0)
    L_train = MPI.Reduce(sum(Ls_train), +, comm, root=0)
    L_train_MAE = MPI.Reduce(sum(Ls_train_MAE), +, comm, root=0)
    L_train_BG = MPI.Reduce(sum(Ls_train_BG), +, comm, root=0)
    forward_time = MPI.Reduce(sum(forward_times), +, comm, root=0)
    backward_time = MPI.Reduce(sum(backward_times), +, comm, root=0) 
    update_time = MPI.Reduce(update_time_local, +, comm, root=0)

    if rank == 0
        #L_train_MAE = L_train_MAE ./ pc_weights_tot
        prof.L_train[batch_id, iter] = L_train ./ pc_weights_tot
        prof.L_train_MAE[batch_id, iter] = L_train_MAE ./ pc_weights_tot
        prof.L_train_BG[batch_id, iter] = L_train_BG ./ pc_weights_tot
        prof.timings[batch_id, iter, 1] = forward_time ./ nranks
        prof.timings[batch_id, iter, 2] = backward_time ./ nranks
        prof.timings[batch_id, iter, 3] = update_time ./ nranks
        if verbosity > 1
            println(" Forward time: $(forward_time ./ nranks) s")
            println(" Backward time: $(backward_time ./ nranks) s")
            println(" Update time: $(update_time ./ nranks) s")
            println(" Learning rate: $(lr_log)")
            #println(" MAE: $L_train_MAE eV")
        end
    end
    if load_sms; unload_simmats!(ham_train, indices); end
end

"""
    val_step!(ham_val, loss, val_data, prof, iter, comm, rank=0)

Evaluates the validation loss for a Hamiltonian model over a given validation dataset, and stores the results in the `HamsterProfiler` instance. This function also tracks the time taken for validation.

# Arguments
- `ham_val`: The Hamiltonian model being validated.
- `loss`: The loss function used to evaluate the performance of the model.
- `val_data`: A collection of validation data.
- `prof`: An instance of the `HamsterProfiler` struct that tracks various profiling information, including validation times and losses.
- `iter`: The current iteration number, used to store the validation results at the correct index in the `prof` instance.

# Returns
- `L_val`: The average validation loss computed over all validation structures. This value is also stored in `prof.L_val` at the index corresponding to `iter`.
- Updates to `prof.val_times`: The elapsed time for the validation step is stored in `prof.val_times[iter]`.
"""
function val_step!(ham_val, losses, val_data, prof, iter, comm; rank=0, nranks=1, valeachiter=valeachiter, conf =get_empty_config())
    ls_weight = get_ls_weight(conf)
    nbatch = get_nbatch(conf)
    val_begin = MPI.Wtime()
    load_sms = nbatch > 1 && !(uses_device_path(ham_val) && val_data isa AbstractVector{<:EigData})
    if load_sms; load_simmats!(ham_val, 1:ham_val.Nstrc); end
    pc_weights = Float64[]
    Ls_val = map(1:ham_val.Nstrc) do index
        push!(pc_weights, losses[index].pc_weight)
        forward(ham_val, index, losses[index], val_data[index])[1] 
    end
    pc_weights_tot = MPI.Allreduce(sum(pc_weights), +, comm)
    Ls_val_sum = ls_weight ? MPI.Allreduce(sum(Ls_val .* pc_weights), +, comm) : MPI.Allreduce(sum(ones(size(Ls_val)) .* pc_weights), +, comm)
    Ls_val_weights = ls_weight ? Ls_val ./ (Ls_val_sum / pc_weights_tot) .* pc_weights : ones(size(Ls_val)) ./ (Ls_val_sum / pc_weights_tot) .* pc_weights
    Ls_val = Ls_val_weights .* Ls_val
    Ls_val_MAE = map(1:ham_val.Nstrc) do index
        forward(ham_val, index, losses[index], val_data[index])[3] * losses[index].pc_weight
    end
    Ls_val_BG = map(1:ham_val.Nstrc) do index
        forward(ham_val, index, losses[index], val_data[index])[4] * losses[index].pc_weight
    end

    all_systems = MPI.gather(ham_val.systems, comm, root=0)
    all_losses = MPI.gather(Ls_val, comm, root=0)
    all_losses_MAE = MPI.gather(Ls_val_MAE, comm, root=0)
    all_losses_BG = MPI.gather(Ls_val_BG, comm, root=0)
    if rank == 0
        all_systems = vcat(all_systems...)
        all_losses = vcat(all_losses...)
        all_losses_MAE = vcat(all_losses_MAE...)
        all_losses_BG = vcat(all_losses_BG...)

        for system in unique(all_systems)
            if !haskey(prof.L_val_system, system)
                prof.L_val_system[system] = zeros(size(prof.L_val))
                prof.L_val_system_MAE[system] = zeros(size(prof.L_val_MAE))
                prof.L_val_system_BG[system] = zeros(size(prof.L_val_BG))
            end
            idxs = findall(s -> s == system, all_systems)
            loss_system = all_losses[idxs]
            prof.L_val_system[system][iter] = mean(loss_system)
            loss_system_MAE = all_losses_MAE[idxs]
            prof.L_val_system_MAE[system][iter] = mean(loss_system_MAE)
            loss_system_BG = all_losses_BG[idxs]
            prof.L_val_system_BG[system][iter] = mean(loss_system_BG)
        end
    end

    val_time_local = MPI.Wtime() - val_begin
    val_time = MPI.Reduce(val_time_local, +, comm, root=0)
    L_val = MPI.Reduce(sum(Ls_val), +, comm, root=0)
    L_val_MAE = MPI.Reduce(sum(Ls_val_MAE), +, comm, root=0)
    L_val_BG = MPI.Reduce(sum(Ls_val_BG), +, comm, root=0)
    if rank == 0
        prof.val_times[iter] = val_time ./ nranks
        prof.L_val[iter-valeachiter+1:iter] .= L_val ./ pc_weights_tot
        prof.L_val_MAE[iter-valeachiter+1:iter] .= L_val_MAE ./ pc_weights_tot
        prof.L_val_BG[iter-valeachiter+1:iter] .= L_val_BG ./ pc_weights_tot
    end
    if load_sms; unload_simmats!(ham_val, 1:ham_val.Nstrc); end
end

"""
    forward(ham::EffectiveHamiltonian, index, loss, data)

Computes the loss for a given Hamiltonian model `ham` using a specified loss function `loss` and input data `data`.
The behavior of the function depends on the type of `data`, which can be either `EigData` or `HrData`.

# Arguments
- `ham::EffectiveHamiltonian`: The Hamiltonian model from which effective Hamiltonians or real-space Hamiltonians are derived.
- `index`: An index that specifies which structure to compute.
- `loss`: A function that calculates the discrepancy between computed and ground truth values.
- `data`: Either an `EigData` object containing k-point and ground truth eigenvalues, or an `HrData` object containing real-space Hamiltonian data.

# Returns
- `L_train::Float64`: The calculated loss.
- `cache`: A preliminary result that is needed to compute the gradient.
"""
function forward(ham::EffectiveHamiltonian, index, data::EigData, l::Loss)
    if uses_device_path(ham); return device_forward(ham, index, data, l); end
    Hk = get_hamiltonian(ham, index, data.kp)
    Es, vs = diagonalize(Hk)

    #offset = mean(Es - data.Es)
    offset = calc_offset(l, data.Es, Es)

    return (Es, vs), offset
end

function forward(Es, loss::Loss, data::EigData)
    L_train = forward(loss, Es, data.Es)
    L_train_MAE = forward_MAE(loss, Es, data.Es)
    L_train_BG = forward_BG(loss, Es, data.Es)
    #println("NZ : $(count(iszero, ham.models[2].params))")
    return L_train, L_train_MAE, L_train_BG
end

function forward(ham::EffectiveHamiltonian, index, loss, data::EigData)
    if uses_device_path(ham); return device_forward(ham, index, loss, data); end
    Hk = get_hamiltonian(ham, index, data.kp)
    Es, vs = diagonalize(Hk)
    L_train = loss(Es, data.Es)
    L_train_MAE = forward_MAE(loss, Es, data.Es)
    L_train_BG = forward_BG(loss, Es, data.Es)
    #println("NZ : $(count(iszero, ham.models[2].params))")
    return L_train, (Es, vs), L_train_MAE, L_train_BG
end

function forward(ham::EffectiveHamiltonian, index, loss, data::HrData)
    Hr = get_hr(ham, index)
    L_train = loss(Hr, data.Hr)
    return L_train, (Hr,)
end

"""
    backward(ham::EffectiveHamiltonian, index, loss, data, cache)

Computes the gradient of the loss for a given Hamiltonian model `ham` with respect to its matrix elements, based on the specified loss function `loss` and input data `data`. 
The function behavior varies depending on the type of `data`, which can be either `EigData` or `HrData`.

# Arguments
- `ham::EffectiveHamiltonian`: The Hamiltonian model for which the gradient of the loss is being computed.
- `index`: An index that specifies which Hamiltonian structure to use in the gradient computation.
- `loss`: A function that calculates the discrepancy between computed and ground truth values.
- `data`: Either an `EigData` object containing k-point and ground truth eigenvalues, or an `HrData` object containing real-space Hamiltonian data.
- `cache`: A preliminary result from `forward` that is required to compute the gradient.

# Returns
- `gradient`: The computed gradient of the loss with respect to the parameters of `ham`.
"""
function backward(L_train_weight::Float64, ham::EffectiveHamiltonian, index, loss, data::EigData, cache, conf=get_empty_config(); nthreads_kpoints=get_nthreads_kpoints(conf), nthreads_bands=get_nthreads_bands(conf))
    if uses_device_path(ham); return device_backward(L_train_weight, ham, index, loss, data, cache); end
    Es_tb, vs = cache
    dL_dE = backward(loss, Es_tb, data.Es) .* L_train_weight
    return get_loss_gradient_hr(dL_dE, vs, ham.Rs[index], data.kp, ham.sp_mode, ham.sp_iterators[index], sp_tol=ham.sp_tol)
end

backward(ham::EffectiveHamiltonian, index, loss, data::HrData, cache, conf) = backward(loss, cache[1], data.Hr)