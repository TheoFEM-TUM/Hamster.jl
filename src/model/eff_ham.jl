struct EffectiveHamiltonian{T, S1, S2}
    Nstrc :: Int64
    models :: T
    sp_mode :: S1
    sp_diag :: S2
    soc :: Bool
    Rs :: Vector{Matrix{Float64}}
end

function EffectiveHamiltonian(conf=get_empty_conf(); mode="pc", index_file="config_inds.dat", tb_model=get_tb_model(conf), sp_mode=get_sp_mode(conf), sp_diag=get_sp_diag(conf), soc=get_soc(conf))
    
    strcs, config_indices = get_structures(conf, mode=mode, index_file=index_file) # TODO: What happens if there is only one Structure? PC/Mixed?
    bases = [Basis(strc, conf) for strc in strcs]
    Rs = [strc.Rs for strc in strcs]

    models = ()
    if tb_model
        models = (models..., TBModel(strcs, bases, conf))
    end

    return EffectiveHamiltonian(length(strcs), models, sp_mode, sp_diag, soc, Rs)
end

"""
    get_hamiltonian(ham::EffectiveHamiltonian, index, ks)

Construct the Hamiltonian matrix for given k-points `ks` from the real-space Hamiltonian and lattice vectors.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object.
- `index::Int`: The index of the structure for which the Hamiltonian is to be calculated.
- `ks`: The k-points for which the Hamiltonian matrix is to be calculated.

# Returns
- `Hk`: The Hamiltonian matrix in reciprocal space corresponding to the given k-points.
"""
function get_hamiltonian(ham::EffectiveHamiltonian, index, ks)
    Hr = get_hr(ham, index)
    Hk = get_hamiltonian(Hr, ham.Rs[index], ks, ham.sp_diag)
    return Hk
end

"""
    get_hr(ham::EffectiveHamiltonian, index)

Retrieve the real-space Hamiltonian (`Hr`) by combining contributions from individual models within the `EffectiveHamiltonian`.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object.
- `index::Int`: The index of the structure for which the Hamiltonian is to be calculated.

# Returns
- `Hr`: The combined real-space Hamiltonian matrix, obtained by summing the Hamiltonians of all the models in the `ham.models` tuple.
"""
function get_hr(ham::EffectiveHamiltonian, index)
    Hr = mapreduce(+, ham.models) do model
        get_hr(model, ham.sp_mode, index, apply_soc=ham.soc)
    end
    return Hr
end

"""
    update!(ham::EffectiveHamiltonian, opt, dL_dHr)

Update the parameters of each model within the `EffectiveHamiltonian` object using a provided update rule.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object containing multiple models.
- `opt`: An optimizer object specifying the update rule (e.g., ADAM).
- `dL_dHr`: The derivative of the loss w.r.t. to each matrix element of the real-space Hamiltonian.

# Returns
- This function modifies the `ham` object in place, updating the parameters of each model it contains.
"""
function update!(ham::EffectiveHamiltonian, opt, reg, dL_dHr)
    for model in ham.models
        update!(model, opt, reg, dL_dHr)
    end
end

"""
    chain_rule(dL_dE, dE_dHr, mode)

Applies the chain rule to compute the gradient of the loss with respect to the real-space Hamiltonian matrix elements by combining partial derivatives `dL_dE` and `dE_dHr`.

# Arguments
- `dL_dE`: An array containing the partial derivatives of the loss with respect to the eigenvalues, with shape `(m, k)`, where `m` is the eigenvalue index and `k` is the k-point index.
- `dE_dHr`: A 3D array of Matrices containing the partial derivatives of the eigenvalues with respect to the Hamiltonian in real-space coordinates, with shape `(R, m, k)`, where `R` is the lattice vector index.
- `mode`: Specifies whether the Hamiltonian is sparse or dense, determining the data structure of the result.

# Returns
- `dL_dHr`: A real-space Hamiltonian gradient array, where each element contains the accumulated gradient for a specific lattice vector in `R`. Its shape is determined by the Hamiltonian structure and `mode`.
"""
function chain_rule(dL_dE, dE_dHr, mode)
    dL_dHr = get_empty_real_hamiltonians(size(dE_dHr, 2), size(dE_dHr, 1), mode)

    for k in axes(dE_dHr, 3)
        for m in axes(dE_dHr, 2)
            for R in axes(dE_dHr, 1)
                @views dL_dHr[R] += dL_dE[m , k] * dE_dHr[R, m, k]
            end
        end
    end
    return dL_dHr
end