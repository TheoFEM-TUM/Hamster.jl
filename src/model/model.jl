"""
    TBModel

A mutable struct representing a tight-binding model.

# Fields
- `h::Matrix{SparseMatrixCSC{Float64, Int64}}`: A matrix where each element is a sparse matrix representing the geometry tensor.
- `parameter_labels::Vector`: A vector containing the label for each parameter.
- `V::Vector{Float64}`: A vector containing the model's parameters.
- `update::Bool`: A boolean flag indicating whether the model's parameters `V` should be updated during optimization or kept fixed.
"""
mutable struct TBModel{G, P}
    hs :: G
    parameter_labels :: P
    V :: Vector{Float64}
    update :: Vector{Bool}
end

"""
    TBModel(strc::Structure, basis::Basis, conf=get_empty_config(); update_tb, initas)

Constructs a `TBModel` for the given structure `strc` and basis `basis`, based on the configuration `conf`.

# Arguments
- `strc::Structure`: The structure of the material or system being modeled.
- `basis::Basis`: The basis functions or orbitals used to describe the electronic states in the tight-binding model.
- `conf`: (Optional) A configuration object that contains various settings and parameters for building the model.
- `update_tb`: (Optional) A flag indicating whether the model's parameters should be updated during optimization. Defaults to `get_update_tb(conf)`.
- `initas`: (Optional) Initialization parameters for the model. Defaults to `get_init_params(conf)`.

# Returns
- A `TBModel` object with the geometry tensor `h` and the model's parameters set via `init_params!`.
"""
function TBModel(strc::Structure, basis::Basis, conf=get_empty_config(); update_tb=get_update_tb(conf, nparams(basis)), initas=get_init_params(conf))
    h = get_geometry_tensor(strc, basis, conf)
    model = TBModel(h, basis.parameters, ones(size(h, 1)), update_tb)
    init_params!(model, basis, conf, initas=initas)
    return model
end

function TBModel(strcs::Vector{Structure}, bases::Vector{<:Basis}, conf=get_empty_config(); update_tb=get_update_tb(conf, nparams(bases[1])), initas=get_init_params(conf))
    hs = map(eachindex(strcs)) do n
        get_geometry_tensor(strcs[n], bases[n], conf)
    end
    model = TBModel(hs, bases[1].parameters, ones(length(update_tb)), update_tb)
    init_params!(model, bases[1], conf, initas=initas)
    return model
end

"""
    get_hr(h, V=model.V, mode=Val{:dense})

Construct the real-space Hamiltonian (`Hr`) by multiplying the geometry tensor `h` with the parameters `V`.

# Arguments
- `h`: A 2D array of matrices where each element `h[v, R]` represents a Hamiltonian block associated with parameter `v` and lattice vector `R`.
- `V`: A vector of parameters.
- `mode`: Optional argument that specifies the format of the resulting Hamiltonian (`Hr`). Defaults to `Val{:dense}`, but can be other types such as sparse.

# Returns
- `Hr`: The resulting real-space Hamiltonian matrix (or array of matrices), constructed by summing the weighted Hamiltonian blocks for each lattice vector `R`.
"""
function get_hr(h::AbstractMatrix, V::AbstractVector, mode=Val{:dense}; apply_soc=false)
    Hr = get_empty_real_hamiltonians(size(h[1, 1], 1), size(h, 2), mode)
    Threads.@threads for R in axes(h, 2)
        for v in axes(h, 1)
            @. Hr[R] += h[v, R] * V[v]
        end
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

get_hr(model::TBModel, mode, index::Int64; apply_soc=false) = get_hr(model.hs[index], model.V, mode, apply_soc=apply_soc)
get_hr(model::TBModel, V, mode; apply_soc=false) = get_hr(model.hs, V, mode, apply_soc=apply_soc)
get_hr(model::TBModel, mode; apply_soc=false) = get_hr(model, model.V, mode, apply_soc=apply_soc)

"""
    update!(model::TBModel, opt, dV)

Updates the parameters of the given TB model `model` using the provided optimizer `opt` and the gradient `dV`.

# Arguments
- `model`: A `TBModel` object that contains the parameters to be updated.
- `opt`: An optimization algorithm or method used to update the model's parameters.
- `dV`: The gradient of the loss w.r.t. to the model parameters.
"""
function update!(model::TBModel, opt, dV)
    update!(opt, model.V, dV)
end

"""
    get_model_gradient(model, dL_dHr)

Computes the gradient of the model's parameters using the given derivative of the Hamiltonian dL_dHr. If dL_dHr is not a vector of matrices (i.e., it represents multiple structures), 
the function assumes it contains gradients for multiple structures and returns the average gradient across these structures.

# Arguments
- `model`: A model object containing the parameters `V` and the geometry tensor `h`.
- `dL_dHr`: A matrix or array representing the derivative of the loss w.r.t. real-space Hamiltonian matrix elements.

# Returns
- The gradients for each parameter in `model.V`.
"""
function get_model_gradient(h, dL_dHr::Vector{<:AbstractMatrix})
    dV = zeros(size(h, 1))
    @tasks for v in axes(h, 1)
        for R in eachindex(dL_dHr)
            dV[v] += sum(h[v, R] .* dL_dHr[R])
        end
    end
    return dV
end

"""
    get_model_gradient(ham::EffectiveHamiltonian, indices, reg, dL_dHr)

Computes the gradient of the loss w.r.t. the model parameters.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian model.
- `indices::AbstractVector`: A set of structure indices.
- `reg`: A regularization term or parameter used in the gradient computation.
- `dL_dHr`: The derivative of the loss function with respect to the Hamiltonian.

# Returns
- `AbstractVector`: A collection of gradients, one for each model in the `ham.models`, computed using the provided indices, regularization term, and loss derivative.
"""
function get_model_gradient(model::TBModel, indices, reg, dL_dHr)
    if any(model.update)
        @views dVs = map(enumerate(indices)) do (n, index)
            get_model_gradient(model.hs[index], dL_dHr[n])
        end
        dV_ = cat(dVs..., dims=2)
        dV_grad = dropdims(sum(dV_, dims=2), dims=2)

        dV_penal = backward(reg, model.V)
        dV = @. ifelse(model.update, dV_grad + dV_penal, 0.)
        return dV
    else
        return zeros(length(model.V))
    end
end

"""
    init_params!(model, basis, conf=get_empty_config(); initas=get_init_params(conf))

Initialize the parameters of a `model` based on the provided configuration and initialization method.

# Arguments
- `model`: The model whose parameter array `V` will be initialized.
- `basis`: The `Basis` structure containing orbital and overlap information, as well as the parameters that will be initialized in the model.
- `conf`: Configuration settings that can be used to customize how parameters are initialized. By default, an empty configuration is used.
- `initas`: Initialization method (e.g., `ones`, `random` or a file).

# Keyword Arguments
- `conf`: Configuration settings.
- `initas`: The initialization method or file path for parameters. Defaults to `ones`.
"""
function init_params!(model, basis, conf=get_empty_config(); initas=get_init_params(conf))
    if initas[1] == 'o'
        set_params!(model, ones(length(model.V)))
    elseif initas[1] == 'r'
        set_params!(model, rand(length(model.V)))
    else
        parameters, parameter_values, _, _, conf_values = read_params(initas)
        check_consistency(conf_values, conf)
        for (v1, basis_param) in enumerate(basis.parameters), (v2, file_param) in enumerate(parameters)
            if basis_param == file_param
                model.V[v1] = parameter_values[v2]
            end
        end
    end
end

"""
    get_params(model::TBModel)

Retrieve the parameters associated with a `TBModel`.

# Arguments
- `model::TBModel`: The tight-binding model instance from which to extract parameters.

# Returns
- The parameters stored in the `V` field of the given `TBModel` instance.
"""
get_params(model::TBModel) = model.V

"""
    write_params(model::TBModel, conf=get_empty_config())

Writes the parameters of a `TBModel` using its parameter labels and values.

# Arguments
- `model::TBModel`: The tight-binding model whose parameters need to be written.
- `conf`: (Optional) Configuration settings, defaults to an empty configuration.
"""
write_params(model::TBModel, conf=get_empty_config()) = write_params(model.parameter_labels, model.V, conf)

"""
    set_params!(model::TBModel, V)

Set the parameters of a `TBModel` instance while ensuring consistency with the model's structure.

# Arguments
- `model::TBModel`: The tight-binding model whose parameters are to be updated.
- `V`: The new parameter vector to assign to the model's `V` field.

# Error Conditions
- Throws an error if the parameter vector `V` is not of the correct size.

# Returns
- Updates the `V` field of the `model` in place if the consistency checks pass.
"""
function set_params!(model::TBModel, V)
    throw_error = size(model.V) ≠ size(V)
    if throw_error
        error("Parameter vector is not of correct size!")
    else
        model.V = V        
    end
end