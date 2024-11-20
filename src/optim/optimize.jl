
function optimize_model!(ham_train, ham_val, optim, dl, conf=get_empty_config(); nbatch=get_nbatch(conf), validate=get_validate(conf))

    for iter in 1:optim.Niter
        for (chunk_id, indices) in enumerate(chunks(1:ham_train.Nstrc, n=nbatch))

            dL_dHr = map(indices) do index
                L_train, cache = forward(ham_train, index, optim.loss, dl.train_data[index])
                @show L_train
                backward(ham_train, index, optim.loss, dl.train_data[index], cache)
            end
            update!(ham_train, indices, optim.adam, optim.reg, dL_dHr)
        end
        if validate
            L_val = mapreduce(+, 1:ham_val.Nstrc) do index
                forward(ham_val, index, optim.loss, dl.val_data[index]) / ham_val.Nstrc
            end
        end
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