"""
    mutable struct SOCModel{M}

A mutable struct representing a model for Spin-Orbit Coupling (SOC) effects.

# Fields:
- `params::Vector{Float64}`: A vector of model parameters.
- `unique_ion_types::Vector{String}`: A vector of unique ion types involved in the system, represented as strings.
- `all_type_types::Vector{String}`: A vector containing all ion types, possibly with repetitions, represented as strings.
- `matrices::Vector{M}`: A vector of matrices that describe the SOC interactions for the given orbital basis.
- `Rs::Matrix{Float64}`: A matrix that contains the lattice translation vectors.
- `update::Bool`: A boolean flag indicating whether the model is set to update its parameters or not.
"""
mutable struct SOCModel{M}
    params :: Vector{Float64}
    unique_ion_types :: Vector{String}
    all_type_types :: Vector{String}
    matrices :: Vector{M}
    Rs :: Matrix{Float64}
    update :: Bool
end

function SOCModel(strc::Structure, basis::Basis, conf=get_empty_config())
    matrices = get_soc_matrices(strc, basis, conf)
    params, unique_ion_types = init_soc_params(strc.ions, conf)
    all_type_types = filter(type->haskey(conf.blocks, type), get_ion_types(strc.ions))
    return SOCModel(params, unique_ion_types, all_type_types, matrices, strc.Rs, get_update_soc(conf))
end

"""
    get_hr(soc_model::SOCModel, sp_mode, index; apply_soc=true) -> BlockDiagonal

Constructs the Hamiltonian representation for a given spin-orbit coupling (SOC) model.

# Arguments
- `soc_model::SOCModel`: The spin-orbit coupling model containing ion types, parameters, and matrices.
- `sp_mode`: Unused parameter (possibly reserved for future functionality).
- `index`: The index of the ion type for parameter retrieval.
- `apply_soc::Bool` (default: `true`): If `true`, applies SOC-related modifications (not explicitly used in this function).

# Returns
- A `BlockDiagonal` matrix where each block corresponds to an ion type, with parameters expanded and applied to SOC matrices.
"""
function get_hr(soc_model::SOCModel, sp_mode, index; apply_soc=true)
    expanded_params = map(soc_model.all_type_types) do ion_type
        index = findfirst(type -> type == ion_type, soc_model.unique_ion_types)
        return soc_model.params[index]
    end
    Msoc = BlockDiagonal(expanded_params .* soc_model.matrices)
    Msoc = convert_block_matrix_to_sparse(Msoc)
    Mzero = spzeros(ComplexF64, size(Msoc, 1), size(Msoc, 2))
    Hr = [ifelse(R⃗ == zeros(3), Msoc, Mzero) for R⃗ in eachcol(soc_model.Rs)]
    return Hr
end

"""
    update!(soc_model::SOCModel, opt, dparams) -> Nothing

Updates the parameters of a spin-orbit coupling (SOC) model using an optimization routine.

# Arguments
- `soc_model::SOCModel`: The SOC model whose parameters will be updated.
- `opt`: The optimizer or update rule to apply to the parameters.
- `dparams`: The parameter updates (e.g., gradients or step values).
"""
function update!(soc_model::SOCModel, opt, dparams)
    update!(opt, soc_model.params, dparams)
end

"""
    get_model_gradient(soc_model::SOCModel, indices, reg, dL_dHr)

Computes the gradient of loss w.r.t. the SOC model's parameters.

# Arguments:
- `soc_model::SOCModel`: The SOC model.
- `indices`: The structure indices (not used here).
- `reg`: A regularization term that penalizes certain parameter values to avoid overfitting or enforce specific behavior.
- `dL_dHr`: Gradient of the loss w.r.t. the Hamiltonian matrix elements.

# Returns:
- `dparams`: A vector containing the gradient of the loss w.r.t. the model parameters.
"""
function get_model_gradient(soc_model::SOCModel, indices, reg, dL_dHr; soc=true)
    iR0 = findR0(soc_model.Rs)
    dparams = zeros(length(soc_model.params))
    if soc_model.update
        for param_index in eachindex(dparams)
            expanded_params = map(soc_model.all_type_types) do ion_type
                index = findfirst(type -> type == ion_type, soc_model.unique_ion_types)
                return ifelse(index == param_index, 1, 0)
            end
            Msoc = BlockDiagonal(expanded_params .* soc_model.matrices)
            Msoc = convert_block_matrix_to_sparse(Msoc)

            for index in indices
                dparams[param_index] += real(sum(dL_dHr[index][iR0] .* Msoc))
            end
        end
    end
    dparams_penal = backward(reg, soc_model.params)
    return dparams + dparams_penal
end

"""
    init_soc_params(ions, conf=get_empty_config(); initas=get_soc_init_params(conf)) -> Vector{Float64}

Initializes spin-orbit coupling (SOC) parameters for a given set of ions based on the specified initialization method.

# Arguments
- `ions`: A collection of ion objects.
- `conf`: (Optional) Configuration object used for consistency checks. Defaults to `get_empty_config()`.
- `initas`: (Optional) Specifies the initialization method:
  - `'z'`: Initializes all parameters to zero.
  - `'o'`: Initializes all parameters to one.
  - `'r'`: Initializes parameters with random values.
  - File path or identifier: Reads SOC parameters from an external source.
"""
function init_soc_params(ions, conf=get_empty_config(); initas=get_soc_init_params(conf))
    unique_ion_types = get_ion_types(ions, uniq=true)
    filter!(type->haskey(conf.blocks, type), unique_ion_types)
    Nparams = length(unique_ion_types)
    if initas[1] == 'z'
        return zeros(Nparams), unique_ion_types
    elseif initas[1] == 'o'
        return ones(Nparams), unique_ion_types
    elseif initas[1] == 'r'
        return rand(Nparams), unique_ion_types
    else
        _, _, unique_ion_types, soc_params, conf_values = read_params(initas)
        check_consistency(conf_values, conf)
        return soc_params, unique_ion_types
    end
end

"""
    get_params(soc_model::SOCModel) -> Vector

Retrieves the parameters of a given spin-orbit coupling (SOC) model.

# Arguments
- `soc_model::SOCModel`: The SOC model containing the parameter set.

# Returns
- A vector of parameters associated with the SOC model.
"""
get_params(soc_model::SOCModel) = soc_model.params

"""
    set_params!(soc_model::SOCModel, params) -> Nothing

Updates the parameters of a spin-orbit coupling (SOC) model.

# Arguments
- `soc_model::SOCModel`: The SOC model whose parameters will be updated.
- `params`: A vector of new parameters to assign to the model.
"""
function set_params!(soc_model::SOCModel, params)
    throw_error = size(soc_model.params) ≠ size(params)
    if throw_error
        error("Parameter vector is not of correct size!")
    else
        soc_model.params = params
    end
end

"""
    write_params(soc_model::SOCModel, conf=get_empty_config()) -> Nothing

Writes the parameters of a spin-orbit coupling (SOC) model to a file.

# Arguments
- `soc_model::SOCModel`: The SOC model whose parameters will be written.
- `conf`: Configuration settings (default: an empty configuration from `get_empty_config()`).
"""
function write_params(soc_model::SOCModel, conf=get_empty_config())
    parameters, parameter_values, _, _, conf_values = read_params("params.dat")
    check_consistency(conf_values, conf)
    write_params(parameters, parameter_values, soc_model.unique_ion_types, soc_model.params, conf)
end