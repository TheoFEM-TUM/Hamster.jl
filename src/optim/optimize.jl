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
function optimize_model!(ham_train, ham_val, optim, dl, prof, comm, conf=get_empty_config(); verbosity=get_verbosity(conf), Nbatch=get_nbatch(conf), validate=get_validate(conf), rank=0, nranks=1)
    print_start_message(prof; verbosity=verbosity)
    for iter in 1:optim.Niter
        for (batch_id, indices) in enumerate(chunks(1:ham_train.Nstrc, n=Nbatch))
            train_step!(ham_train, indices, optim, dl.train_data, prof, iter, batch_id, comm, conf, rank=rank, nranks=nranks)
            print_train_status(prof, iter, batch_id, verbosity=verbosity)
        end
        if validate
            print_val_start(prof, iter, verbosity=verbosity)
            copy_params!(ham_val, ham_train)
            val_step!(ham_val, optim.val_loss, dl.val_data, prof, iter, comm, rank=rank, nranks=nranks)
            print_val_status(prof, iter, verbosity=verbosity)
        end
        MPI.Barrier(comm)
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
function train_step!(ham_train, indices, optim, train_data, prof, iter, batch_id, comm, conf=get_empty_config(); rank=0, nranks=1)
    Nstrc_tot = MPI.Reduce(length(indices), +, comm, root=0)
    forward_times = Float64[]
    backward_times = Float64[]
    Ls_train = Float64[]

    dL_dHr = map(indices) do index
        f_time = @elapsed L_train, cache = forward(ham_train, index, optim.loss, train_data[index])
        b_time = @elapsed dL_dHr_index = backward(ham_train, index, optim.loss, train_data[index], cache, conf)
        push!(forward_times, f_time); push!(backward_times, b_time); push!(Ls_train, L_train)
        return dL_dHr_index
    end

    update_begin = MPI.Wtime()
    for model in ham_train.models
        model_grad_local = get_model_gradient(model, indices, optim.reg, dL_dHr)
        model_grad = MPI.Reduce(model_grad_local, +, comm, root=0)
        if rank == 0; update!(model, optim.adam, model_grad ./ Nstrc_tot); end
        params = get_params(model)
        MPI.Bcast!(params, comm, root=0)
        set_params!(model, params)
    end
    update_time_local = MPI.Wtime() - update_begin

    L_train = MPI.Reduce(sum(Ls_train), +, comm, root=0)
    forward_time = MPI.Reduce(sum(forward_times), +, comm, root=0)
    backward_time = MPI.Reduce(sum(backward_times), +, comm, root=0) 
    update_time = MPI.Reduce(update_time_local, +, comm, root=0)

    if rank == 0
        prof.L_train[batch_id, iter] = L_train ./ Nstrc_tot
        prof.timings[batch_id, iter, 1] = forward_time ./ nranks
        prof.timings[batch_id, iter, 2] = backward_time ./ nranks
        prof.timings[batch_id, iter, 3] = update_time ./ nranks
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
function val_step!(ham_val, loss, val_data, prof, iter, comm; rank=0, nranks=1)
    Nstrc_tot = MPI.Reduce(ham_val.Nstrc, +, comm, root=0)
    val_time = @elapsed begin 
        L_val = mapreduce(+, 1:ham_val.Nstrc) do index
            forward(ham_val, index, loss, val_data[index])[1] / ham_val.Nstrc
        end
    end
    MPI.Reduce(L_val, +, comm, root=0)
    if rank == 0
        prof.val_times[iter] = val_time ./ nranks
        prof.L_val[iter] = L_val ./ Nstrc_tot
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
function forward(ham::EffectiveHamiltonian, index, loss, data::EigData)
    Hk = get_hamiltonian(ham, index, data.kp)
    Es, vs = diagonalize(Hk)
    L_train = loss(Es, data.Es)
    return L_train, (Es, vs)
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
function backward(ham::EffectiveHamiltonian, index, loss, data::EigData, cache, conf=get_empty_config(); nthreads_kpoints=get_nthreads_kpoints(conf), nthreads_bands=get_nthreads_bands(conf))
    Es_tb, vs = cache
    dL_dE = backward(loss, Es_tb, data.Es)
    dE_dHr = get_eigenvalue_gradient(vs, ham.Rs[index], data.kp, ham.sp_mode, ham.sp_iterator, nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands, sp_tol=ham.sp_tol)
    dL_dHr = chain_rule(dL_dE, dE_dHr, ham.sp_mode, nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands, sp_tol=ham.sp_tol)
    return dL_dHr
end

backward(ham::EffectiveHamiltonian, index, loss, data::HrData, cache, conf) = backward(loss, cache[1], data.Hr)