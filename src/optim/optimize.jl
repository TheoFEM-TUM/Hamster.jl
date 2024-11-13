
function optimize_model!(ham_train, ham_val, optim, dl, conf; nbatch=get_nbatch(conf))

    for iter in 1:Niter
        for (chunk_id, indices) in enumerate(chunks(1:ham_train.Nstrc, n=nbatch))

            dL_dHr = map(indices) do index
                L_train, cache = forward(ham_train, index, loss, dl.train_data[index])
                backward(ham_train, index, loss, dl.train_data[index], cache)
            end
            update!(ham, indices, optim.opt, optim.reg, dL_dHr)
        end
        L_val = mapreduce(+, 1:ham_val.Nstrc) do index
            ks, ground_truth = dl.val_data[index]
            forward(ham_val, index, loss, ground_truth) / ham_val.Nstrc
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
function forward(ham::EffectiveHamiltonian, index, loss, train_data::EigData)
    ks, ground_truth = train_data
    Hk = get_hamiltonian(ham, index, ks)
    Es, vs = diagonalize(Hk)
    L_train = loss(Es, ground_truth)
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
function backward(ham::EffectiveHamiltonian, index, loss, data::EigData, cache)
    Es_tb, vs = cache
    dL_dE = backward(loss, Es_tb, data.Es)
    dE_dHr = get_eigenvalue_gradient(vs, ham_train.Rs[index], data.ks)
    return chain_rule(dL_dE, dE_dHr, ham.mode)
end

backward(ham::EffectiveHamiltonian, index, loss, data::EigData, cache) = backward(loss, cache[1], data.Hr)