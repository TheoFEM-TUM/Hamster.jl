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

# Workflow
1. Print the start message.
2. For each training iteration, split the training data into batches and perform training steps.
3. Optionally validate the model after each training iteration.
4. Print the final status once training is complete.

# Returns
- Updates the HamsterProfiler `prof` and the model parameters in `ham_train` and `ham_val`.
"""
function optimize_model!(ham_train, ham_val, optim, dl, prof, comm, conf=get_empty_config(); verbosity=get_verbosity(conf), Nbatch=get_nbatch(conf), validate=get_validate(conf), valeachiter=get_valeachiter(conf), rank=0, nranks=1)
    print_start_message(prof; verbosity=verbosity)
    for iter in 1:optim.Niter
        iter_begin = MPI.Wtime()
        for (batch_id, indices) in enumerate(chunks(1:ham_train.Nstrc, n=Nbatch))
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
        MPI.Barrier(comm)
        iter_time = MPI.Wtime() - iter_begin
        if verbosity > 1 && rank == 0; println("Iteration time: $iter_time s"); end
    end
    print_final_status(prof; verbosity=verbosity)
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
    ls_weight = get_ls_weight(conf)
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
    for model in ham_train.models
        model_grad_local = get_model_gradient(model, indices, optim.reg, dL_dHr; soc=ham_train.soc)
        model_grad = MPI.Reduce(model_grad_local, +, comm, root=0)
        if rank == 0; update!(model, optim.adam, model_grad ./ pc_weights_tot); end
        params = get_params(model)
        #if rank == 0; @info "NZ params $(count(iszero, params))" ; end
        #if rank == 0; @info "NZ grad params $(count(iszero, model_grad))" ; end
        MPI.Bcast!(params, comm, root=0)
        set_params!(model, params)
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
    val_begin = MPI.Wtime()
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
    Es_tb, vs = cache
    dL_dE = backward(loss, Es_tb, data.Es) .* L_train_weight
    dE_dHr = get_eigenvalue_gradient(vs, ham.Rs[index], data.kp, ham.sp_mode, ham.sp_iterators[index], nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands, sp_tol=ham.sp_tol)
    dL_dHr = chain_rule(dL_dE, dE_dHr, ham.sp_mode, nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands, sp_tol=ham.sp_tol)
    return dL_dHr
end

backward(ham::EffectiveHamiltonian, index, loss, data::HrData, cache, conf) = backward(loss, cache[1], data.Hr)