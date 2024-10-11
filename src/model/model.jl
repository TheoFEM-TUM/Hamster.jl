"""
    TBModel

A mutable struct representing a tight-binding model.

# Fields
- `h::Matrix{SparseMatrixCSC{Float64, Int64}}`: A matrix where each element is a sparse matrix representing the geometry tensor.
- `V::Vector{Float64}`: A vector containing the model's parameters.
- `update::Bool`: A boolean flag indicating whether the model's parameters `V` should be updated during optimization or kept fixed.
"""
mutable struct TBModel
    h :: Matrix{SparseMatrixCSC{Float64, Int64}}
    V :: Vector{Float64}
    update :: Bool
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
function TBModel(strc::Structure, basis::Basis, conf=get_empty_config(); update_tb=get_update_tb(conf), initas=get_init_params(conf))
    h = get_geometry_tensor(strc, basis, conf)
    model = TBModel(h, ones(size(h, 1)), update_tb)
    init_params!(model, basis, conf, initas=initas)
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
function get_hr(model::TBModel, V::AbstractVector, mode=Val{:dense}; apply_soc=false)
    Hr = get_empty_real_hamiltonians(size(model.h[1, 1], 1), size(model.h, 2), mode)
    Threads.@threads for R in axes(model.h, 2)
        for v in axes(model.h, 1)
            @. Hr[R] += model.h[v, R] * V[v]
        end
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

get_hr(model::TBModel, mode; apply_soc=false) = get_hr(model, model.V, mode, apply_soc=apply_soc)

"""
    update!(model::TBModel, opt, dL_dHr)

Updates the parameters of the given tight-binding model `model` using the provided optimization method `opt` and the derivative of the loss function with respect to the Hamiltonian `dL_dHr`.

# Arguments
- `model`: A `TBModel` object that contains the parameters to be updated.
- `opt`: An optimization algorithm or method used to update the model's parameters.
- `dL_dHr`: A matrix or array representing the derivative of the loss function with respect to the Hamiltonian.
"""
function update!(model::TBModel, opt, dL_dHr)
    if model.update    
        dV = get_model_gradient(model, dL_dHr)
        update!(model.V, opt, dV)
    end
end

"""
    get_model_gradient(model, dHr)

Calculates the gradient of the model's parameters based on the provided derivative of the Hamiltonian `dHr`.

# Arguments
- `model`: A model object containing the parameters `V` and the geometry tensor `h`.
- `dHr`: A matrix or array representing the derivative of the loss w.r.t. real-space Hamiltonian matrix elements.

# Returns
- The gradients for each parameter in `model.V`.
"""
function get_model_gradient(model, dL_dHr)
    dV = zeros(length(model.V))
    for v in axes(model.h, 1)
        for R in eachindex(dHr)
            dV[v] += sum(model.h[v, R] * dL_dHr[R])
        end
    end
    return dV
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
        for v in eachindex(model.V)
            model.V[v] = 1.
        end
    elseif initas[1] == 'r'
        for v in eachindex(model.V)
            model.V[v] = rand()
        end
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

