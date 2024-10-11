struct EffectiveHamiltonian{T, S1, S2}
    models :: T
    sp_mode :: S1
    sp_diag :: S2
    soc :: Bool
    Rs :: Matrix{Int64}
end

function EffectiveHamiltonian(conf=get_empty_conf(); tb_model=get_tb_model(conf), sp_mode=get_sp_mode(conf), sp_diag=get_sp_diag(conf), soc=get_soc(conf))
    strc = Structure(conf)
    basis = Basis(strc, conf)
    
    models = ()
    if tb_model
        models = (models..., TBModel(strc, basis, conf))
    end

    return EffectiveHamiltonian(models, sp_mode, sp_diag, soc, strc.Rs)
end

"""
    (ham::EffectiveHamiltonian)(ks)

Compute the eigenvalues and eigenvectors (diagonalization) of the effective Hamiltonian matrix at k-points `ks`.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object containing the system's Hamiltonian data.
- `ks`: The k-points at which the Hamiltonian matrix should be evaluated.

# Returns
- A tuple `(eigenvalues, eigenvectors)`.
"""
function (ham::EffectiveHamiltonian)(ks)
    Hk = get_hamiltonian(ham, ks)
    return diagonalize(Hk)
end

"""
    get_hamiltonian(ham::EffectiveHamiltonian, ks)

Construct the Hamiltonian matrix for given k-points `ks` from the real-space Hamiltonian and lattice vectors.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object.
- `ks`: The k-points for which the Hamiltonian matrix is to be calculated.

# Returns
- `Hk`: The Hamiltonian matrix in reciprocal space corresponding to the given k-points.
"""
function get_hamiltonian(ham::EffectiveHamiltonian, ks)
    Hr = get_hr(ham)
    Hk = get_hamiltonian(Hr, ham.Rs, ks, ham.sp_diag)
    return Hk
end

"""
    get_hr(ham::EffectiveHamiltonian)

Retrieve the real-space Hamiltonian (`Hr`) by combining contributions from individual models within the `EffectiveHamiltonian`.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object.

# Returns
- `Hr`: The combined real-space Hamiltonian matrix, obtained by summing the Hamiltonians of all the models in the `ham.models` tuple.
"""
function get_hr(ham::EffectiveHamiltonian)
    Hr = mapreduce(+, ham.models) do model
        get_hr(model, ham.sp_mode, apply_soc=ham.soc)
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
function update!(ham::EffectiveHamiltonian, opt, dL_dHr)
    for model in ham.models
        update!(model, opt, dL_dHr)
    end
end

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