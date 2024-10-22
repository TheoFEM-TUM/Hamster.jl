
function optimize_model!(ham_train, ham_val, optim, conf; nbatch=get_nbatch(conf))

    for iter in 1:Niter
        for (chunk_id, indices) in enumerate(chunks(1:ham_train.Nstrc, n=nbatch))

            dL_dHr = map(indices) do index
                L_train, cache = forward(ham_train, index, loss, train_data[index], fit_eigenvalues=fit_eigenvalues)
                backward(ham_train, index, loss, ground_truth, cache, fit_eigenvalues=fit_eigenvalues)
            end
            update!(ham, indices, optim.opt, optim.reg, dL_dHr)
        end
        L_val = mapreduce(+, 1:ham_val.Nstrc) do index
            ks, ground_truth = val_data[index]
            forward(ham_val, index, loss, ground_truth, fit_eigenvalues=fit_eigenvalues) / ham_val.Nstrc
        end
    end
end

function forward(ham::EffectiveHamiltonian, index, loss, train_data; fit_eigenvalues=true)
    ks, ground_truth = train_data
    if fit_eigenvalues
        Hk = get_hamiltonian(ham, index, ks)
        Es, vs = diagonalize(Hk)
        L_train = loss(Es, ground_truth)
        return L_train, (E, vs)
    else
        Hr = get_hr(ham, index)
        L_train = loss(Hr, ground_truth)
        return L_train, Hr
    end
end

function backward(ham::EffectiveHamiltonian, index, loss, ground_truth, cache; fit_eigenvalues=true)
    if fit_eigenvalues
        Es, vs = cache
        dL_dE = backward(loss, Es, ground_truth)
        dE_dHr = get_eigenvalue_gradient(vs, ham_train.Rs[index], ks)
        return chain_rule(dL_dE, dE_dHr, ham.mode)
    else
        return backward(loss, cache, ground_truth)
    end
end