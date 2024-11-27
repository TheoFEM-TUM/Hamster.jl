struct EffectiveHamiltonian{T, S1, S2}
    Nstrc :: Int64
    models :: T
    sp_mode :: S1
    sp_diag :: S2
    sp_tol :: Float64
    soc :: Bool
    Rs :: Vector{Matrix{Float64}}
end

function EffectiveHamiltonian(strcs, bases, conf=get_empty_conf(); mode="pc", index_file="config_inds.dat", tb_model=get_tb_model(conf), sp_mode=get_sp_mode(conf), sp_diag=get_sp_diag(conf), sp_tol = get_sp_tol(conf), soc=get_soc(conf))
    if isempty(strcs) && isempty(bases)
        return EffectiveHamiltonian(0, nothing, Dense(), Dense(), 1e-10, false, [zeros(3, 1)])
    end
    
    Rs = [strc.Rs for strc in strcs]

    models = ()
    if tb_model
        models = (models..., TBModel(strcs, bases, conf))
    end

    return EffectiveHamiltonian(length(strcs), models, sp_mode, sp_diag, sp_tol, soc, Rs)
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
- `model_grad`: The gradient of the loss w.r.t. the model parameters.

# Returns
- This function modifies the `ham` object in place, updating the parameters of each model it contains.
"""
function update!(ham::EffectiveHamiltonian, opt, model_grad)
    for (model, grad) in zip(ham.models, model_grad)
        update!(model, opt, grad)
    end
end

"""
    copy_params!(receiving_ham::H1, sending_ham::H2) where {H1,H2<:EffectiveHamiltonian}

Copy parameters from one EffectiveHamiltonian (`sending_ham`) to another (`receiving_ham`).

# Arguments
- `receiving_ham`: The Hamiltonian object that will receive the parameters.
- `sending_ham`: The Hamiltonian object providing the parameters to be copied.
"""
function copy_params!(receiving_ham::H1, sending_ham::H2) where {H1,H2<:EffectiveHamiltonian}
    for (receiving_model, sending_model) in zip(receiving_ham.models, sending_ham.models) 
        set_params!(receiving_model, get_params(sending_model))
    end
end