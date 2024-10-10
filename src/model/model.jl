mutable struct TBModel
    h :: Matrix{SparseMatrixCSC{Float64, Int64}}
    V :: Vector{Float64}
    update :: Bool
end

function TBModel(strc::Structure, basis::Basis, conf=get_empty_config(); update_tb=get_update_tb(conf))
    h = get_geometry_tensor(strc, basis, conf)
    V = ones(size(h, 1))
    return TBModel(h, V, update_tb)
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

function update!(model::TBModel, opt, dHr)
    if model.update    
        dV = zeros(length(model.V))
        for v in axes(model.h, 1)
            for R in eachindex(dHr)
                dV[v] += sum(model.h[v, R] * dHr[R])
            end
        end
        update!(model.V, opt, dV)
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

