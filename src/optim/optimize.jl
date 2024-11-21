"""
    optimize_model!(ham_train, ham_val, optim, dl, prof, conf=get_empty_config(); verbosity=get_verbosity(conf), Nbatch=get_nbatch(conf), validate=get_validate(conf))

Optimizes the model by performing training and optional validation steps.

# Arguments
- `ham_train`: The Hamiltonian model used for training.
- `ham_val`: The Hamiltonian model used for validation (optional).
- `optim`: An optimization configuration, including the optimizer and its settings.
- `dl`: A data loader object containing the training data.
- `prof`: A profiler object used to store training and validation information.
- `conf`: A configuration object containing additional settings (default is an empty config).
- `verbosity`: The level of verbosity for logging (default is set by `get_verbosity(conf)`).
- `Nbatch`: The number of batches per training iteration (default is set by `get_nbatch(conf)`).
- `validate`: A flag indicating whether to perform validation during training (default is set by `get_validate(conf)`).

# Description
This function optimizes a model by iterating through training steps and optionally validating the model after each training iteration. It reports the progress of training and validation via printing functions at each iteration. The training step involves computing the loss, performing backpropagation, and updating the model parameters. If `validate` is set to true, the model is evaluated on a validation dataset after each training iteration.

# Workflow
1. Print the start message.
2. For each training iteration, split the training data into batches and perform training steps.
3. Optionally validate the model after each training iteration.
4. Print the final status once training is complete.

# Returns
- This function does not return any value but updates the `prof` object with training and validation statistics and updates model parameters.
"""
function optimize_model!(ham_train, ham_val, optim, dl, prof, conf=get_empty_config(); verbosity=get_verbosity(conf), Nbatch=get_nbatch(conf), validate=get_validate(conf))
    print_start_message(prof; verbosity=verbosity)
    for iter in 1:optim.Niter
        for (batch_id, indices) in enumerate(chunks(1:ham_train.Nstrc, n=Nbatch))
            train_step!(ham_train, indices, optim, dl.train_data, prof, iter, batch_id)
            print_train_status(prof, iter, batch_id, verbosity=verbosity)
        end
        if validate
            # TODO: ham_val currently gets no parameter information
            print_val_start(verbosity=verbosity)
            val_step!(ham_val, optim.loss, val_data, prof, iter)
            print_val_status(prof, iter, verbosity=verbosity)
        end
    end
    print_final_status(prof; verbosity=verbosity)
end

"""
train_step!(ham_train, indices, optim, train_data)

Performs a single training step on a Hamiltonian model by computing gradients and updating model parameters.

# Arguments
- `ham_train`: The Hamiltonian model being trained.
- `indices`: A collection of indices specifying which training data points to process in this step.
- `optim`: An object encapsulating optimization parameters, such as the loss function, regularization, and optimizer.
- `train_data`: A collection of training data corresponding to the indices. Each entry contains the input-output pairs or features for training.

# Workflow
1. Iterates through the given `indices` to compute the loss (`L_train`) and cache intermediate values using the `forward` function.
2. Calls `backward` to compute the gradient of the loss with respect to the model parameters (`dL_dHr`) for each index.
3. Updates the model parameters using the computed gradients and the specified optimizer via `update!`.

# Side Effects
- Updates the model parameters in-place within `ham_train`.
"""
function train_step!(ham_train, indices, optim, train_data, prof, iter, batch_id)
    forward_times = Float64[]
    backward_times = Float64[]
    Ls_train = Float64[]
    dL_dHr = map(indices) do index
        forward_time = @elapsed L_train, cache = forward(ham_train, index, optim.loss, train_data[index])
        backward_time = @elapsed dL_dHr_index = backward(ham_train, index, optim.loss, train_data[index], cache)
        push!(forward_times, forward_time); push!(backward_times, backward_time); push!(Ls_train, L_train)
        return dL_dHr_index
    end
    update_time = @elapsed update!(ham_train, indices, optim.adam, optim.reg, dL_dHr)
    prof.L_train[batch_id, iter] = mean(Ls_train)
    prof.timings[batch_id, iter, 1] = sum(forward_times)
    prof.timings[batch_id, iter, 2] = sum(backward_times)
    prof.timings[batch_id, iter, 3] = update_time
end

"""
val_step!(ham_val, loss, val_data, prof, iter)

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
function val_step!(ham_val, loss, val_data, prof, iter)
    val_time = @elapsed L_val = mapreduce(+, 1:ham_val.Nstrc) do index
        forward(ham_val, index, loss, val_data[index])[1] / ham_val.Nstrc
    end
    prof.val_times[iter] = val_time
    prof.L_val[iter] = L_val
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
    vs_out = reshape_and_sparsify_eigenvectors(vs, ham.sp_mode)
    L_train = loss(Es, data.Es)
    return L_train, (Es, vs_out)
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
function backward(ham::EffectiveHamiltonian, index, loss, data::EigData, cache)
    Es_tb, vs = cache
    dL_dE = backward(loss, Es_tb, data.Es)
    dE_dHr = get_eigenvalue_gradient(vs, ham.Rs[index], data.kp)
    return chain_rule(dL_dE, dE_dHr, ham.sp_mode)
end

backward(ham::EffectiveHamiltonian, index, loss, data::HrData, cache) = backward(loss, cache[1], data.Hr)