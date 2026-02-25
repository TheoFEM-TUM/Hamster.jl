"""
    mutable struct SOCModel{V, M}

A mutable struct representing a model for Spin-Orbit Coupling (SOC) effects.

# Type Parameters
- `V`: Type representing a structure-specific ion labeling or identifier.
- `M`: Type of the matrices describing SOC interactions, e.g., `Matrix{ComplexF64}`.

# Fields
- `params::Vector{Float64}`: A vector of model parameters.
- `param_labels::Vector{UInt8}`: A vector of integer labels identifying which parameters belong to which ion.
- `types_per_strc::Vector{V}`: A vector containing the ion labeling or types for each structure in the dataset.
- `matrices::OrderedDict{UInt8, M}`: Ordered dictionary mapping integer type keys to SOC matrices for the orbital basis.
- `Rs_info::Matrix{Int64}`: Matrix storing R0 and number of translation vectors for each structure.
- `update::Bool`: Flag indicating whether the model is currently set to update its parameters.
"""
mutable struct SOCModel{V, M}
    params :: Vector{Float64}
    param_labels :: Vector{UInt8}
    types_per_strc :: Vector{V}
    matrices :: OrderedDict{UInt8, M}
    Rs_info :: Matrix{Int64}
    update :: Bool
end

function SOCModel(strcs::Vector{Structure}, bases::Vector{<:Basis}, comm, conf=get_empty_config(); 
        rank=0,
        update_soc=get_soc_update(conf))
    
    soc_matrices_per_type = OrderedDict{UInt8, Matrix{ComplexF64}}()
    types_per_strc = Vector{UInt8}[]
    Rs_info = zeros(Int64, 2, length(strcs))
    
    for n in eachindex(strcs)
        types_for_strc = filter(type->haskey(conf.blocks, number_to_element(type)), [ion.type for ion in strcs[n].ions])
        push!(types_per_strc, types_for_strc)

        soc_matrices = get_soc_matrices(strcs[n], bases[n], conf)
        merge_soc_matrices!(soc_matrices_per_type, soc_matrices)

        Rs_info[1, n] = size(strcs[n].Rs, 2)
        Rs_info[2, n] = findR0(strcs[n].Rs)
    end

    param_labels_local = UInt8.(collect(keys(soc_matrices_per_type)))
    param_labels = MPI.gather(param_labels_local, comm, root=0)
    if rank == 0
        param_labels = unique(Iterators.flatten(param_labels))
        nparams = length(param_labels)
    else
        param_labels = UInt8[]
        nparams = 0
    end
    nparams = MPI.Bcast(nparams, 0, comm)

    if rank ≠ 0
        resize!(param_labels, nparams)
    end
    MPI.Bcast!(param_labels, comm, root=0)
    MPI.Barrier(comm)

    params = init_soc_params(param_labels, conf)
    return SOCModel(params, param_labels, types_per_strc, soc_matrices_per_type, Rs_info, update_soc)
end

get_param(soc_model::SOCModel, type) = soc_model.params[findfirst(t->t==type, soc_model.param_labels)]

check_param_type(soc_model, param_index, type) = findfirst(k->k==type, soc_model.param_labels) == param_index

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
    Msoc = BlockDiagonal([get_param(soc_model, type) * soc_model.matrices[type] for type in soc_model.types_per_strc[index]])
    Msoc = convert_block_matrix_to_sparse(Msoc)
    Mzero = spzeros(ComplexF64, size(Msoc, 1), size(Msoc, 2))
    Hr = [ifelse(R == soc_model.Rs_info[2, index], Msoc, Mzero) for R in 1:soc_model.Rs_info[1, index]]
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
    dparams = zeros(length(soc_model.params))
    if soc_model.update
        for index in indices, param_index in eachindex(dparams)
            iR0 = soc_model.Rs_info[2, index]
            Msoc = BlockDiagonal([ifelse(check_param_type(soc_model, param_index, type), 1, 0) .* soc_model.matrices[type]
                                for type in soc_model.types_per_strc[index]])
            Msoc = convert_block_matrix_to_sparse(Msoc)
            dparams[param_index] += real(sum(dL_dHr[index][iR0] .* Msoc))
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
function init_soc_params(ion_types, conf=get_empty_config(); initas=get_soc_init_params(conf))
    Nparams = length(ion_types)
    if initas[1] == 'z'
        return zeros(Nparams)
    elseif initas[1] == 'o'
        return ones(Nparams)
    elseif initas[1] == 'r'
        return rand(Nparams)
    else
        params = zeros(Nparams)
        _, _, unique_ion_types, soc_params, conf_values = read_params(initas)
        check_consistency(conf_values, conf)
        for (n, type) in enumerate(element_to_number.(unique_ion_types))
            params[findfirst(t->type==t, ion_types)] = soc_params[n]
        end

        return params
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
    write_params(parameters, parameter_values, number_to_element.(soc_model.param_labels), soc_model.params, conf)
end