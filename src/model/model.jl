"""
    RadialExponents

Additional distance dependence `(d / Å)^(-c)` of the TB parameters (tag `radial_exponent`): one exponent `c` per parameter
(fixed at 0 for on-site parameters), whether the exponents are fitted (`update`), and per structure the nonzeros of the
geometry tensor without the factor (`h0[n][v, R]`) and the logarithms `ln(d / Å)` of their bond lengths (`logr[n][v, R]`),
both in the order of `nonzeros(hs[n][v, R])`.
"""
mutable struct RadialExponents
    c :: Vector{Float64}
    update :: Bool
    h0 :: Vector{Matrix{Vector{Float64}}}
    logr :: Vector{Matrix{Vector{Float64}}}
end

"""
    TBModel

A mutable struct representing a tight-binding model.

# Fields
- `h::Matrix{SparseMatrixCSC{Float64, Int64}}`: A matrix where each element is a sparse matrix representing the geometry tensor.
- `parameter_labels::Vector`: A vector containing the label for each parameter.
- `V::Vector{Float64}`: A vector containing the model's parameters.
- `update::Bool`: A boolean flag indicating whether the model's parameters `V` should be updated during optimization or kept fixed.
- `radial`: The radial exponents (`RadialExponents`, tag `radial_exponent`) or `nothing`. With exponents, `hs` contains
  the geometry tensor including the factors `d^(-c)`.
"""
mutable struct TBModel{G, P}
    hs :: G
    params :: Vector{Float64}
    param_labels :: Vector{P}
    params_per_strc :: Vector{Vector{Int64}}
    update :: Vector{Bool}
    radial :: Union{Nothing, RadialExponents}
end

TBModel(hs, params, param_labels, params_per_strc, update) = TBModel(hs, params, param_labels, params_per_strc, update, nothing)

"""
    has_exponents(model)

Whether `model` is a TB model with radial exponents (see `RadialExponents`).
"""
has_exponents(model) = model isa TBModel && !isnothing(model.radial)

"""
    hopping_parameters(model::TBModel)

Indices of the parameters between different atoms (nearest-neighbor label ≠ 0), the parameters with a radial exponent.
"""
hopping_parameters(model::TBModel) = findall(p -> p.nnlabel ≠ 0, model.param_labels)

"""
    add_radial_exponents!(model, logr, conf; update=get_update_radial_exponent(conf), initas=get_init_params(conf))

Add radial exponents (see `RadialExponents`) to `model`, whose geometry tensors do not contain them yet, with the bond
lengths `logr` from `get_geometry_tensor(...; log_distances=true)`. The exponents start from 0 or from the
`exponent_<label>` lines of the parameter file `initas`.
"""
function add_radial_exponents!(model, logr, conf; update=get_update_radial_exponent(conf), initas=get_init_params(conf))
    h0 = [map(h -> copy(nonzeros(h)), hs) for hs in model.hs]
    model.radial = RadialExponents(zeros(length(model.params)), update, h0, logr)
    if get_init_mode(initas) == :file
        labels, values = read_params(initas)[10:11]
        file_values = Dict(zip(labels, values))
        for v in hopping_parameters(model)
            model.radial.c[v] = get(file_values, model.param_labels[v], 0.0)
        end
    end
    apply_radial_exponents!(model)
    return model
end

"""
    base_geometry_tensor(model::TBModel, n)

Geometry tensor of structure `n` without the factors of the radial exponents, e.g. for the ML descriptors, which thus do
not depend on the exponents.
"""
function base_geometry_tensor(model::TBModel, n)
    has_exponents(model) || return model.hs[n]
    return map((h, h0) -> SparseMatrixCSC(h.m, h.n, h.colptr, h.rowval, copy(h0)), model.hs[n], model.radial.h0[n])
end

"""
    apply_radial_exponents!(model::TBModel)

Recompute the geometry tensors of `model` with the current radial exponents: `h[v, R] = h0[v, R] (d / Å)^(-c_v)`.
"""
function apply_radial_exponents!(model::TBModel)
    has_exponents(model) || return model
    c = model.radial.c
    for n in eachindex(model.hs)
        tforeach(CartesianIndices(model.hs[n])) do I
            cv = c[model.params_per_strc[n][I[1]]]
            nonzeros(model.hs[n][I]) .= model.radial.h0[n][I] .* exp.(-cv .* model.radial.logr[n][I])
        end
    end
    return model
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
- A `TBModel` object with the geometry tensor `h` and the model's parameters set via `init_params!`. Radial exponents
  (`radial_exponent`) need the constructor for a vector of structures.
"""
function TBModel(strc::Structure, basis::Basis, conf=get_empty_config(); update_tb=get_update_tb(conf, nparams(basis)), initas=get_init_params(conf))
    get_radial_exponent(conf) && error("radial_exponent = true needs TBModel(strcs::Vector{Structure}, bases, comm, conf).")
    h = get_geometry_tensor(strc, basis, conf)
    Nparams = length(basis.parameters)
    model = TBModel(h, ones(Nparams), basis.parameters, [collect(1:Nparams)], update_tb)
    init_params!(model, conf, initas=initas)
    return model
end

function TBModel(strcs::Vector{Structure}, bases::Vector{<:Basis}, comm, conf=get_empty_config();
                rank=0,
                nranks=1,
                update_tb=nothing,
                filename = get_rllm_file(conf),
                initas=get_init_params(conf),
                label="",
                write_summary=true)
                
    if get_load_rllm(conf) == false
        rllm_file = get_rllm_file(conf)
        if isfile(rllm_file) && rank == 0; rm(rllm_file); end
        precalc_rllm(bases, conf, rank=rank, nranks=nranks, comm=comm)
    end
    file = nothing
    if occursin(".h5", filename)
        if isnothing(comm)
            file = h5open(filename, "r")
        else
            file = h5open(filename, "r", comm)
        end
    end
    radial = get_radial_exponent(conf)
    tensors = map(eachindex(strcs)) do n
        get_geometry_tensor(strcs[n], bases[n], conf, file=file, comm=comm, log_distances=radial)
    end
    hs = radial ? first.(tensors) : tensors
    if occursin(".h5", filename)
        close(file)
    end

    param_labels_local = unique(Iterators.flatten([basis.parameters for basis in bases]))
    param_labels = MPI.gather(param_labels_local, comm, root=0)
    if rank == 0
        param_labels = unique(Iterators.flatten(param_labels))
        nparams = length(param_labels)
    else
        param_labels = Vector{ParameterLabel}()
        nparams = 0
    end
    nparams = MPI.Bcast(nparams, 0, comm)

    if rank ≠ 0
        resize!(param_labels, nparams)
    end
    MPI.Bcast!(param_labels, comm, root=0)
    MPI.Barrier(comm)

    # Indices in update_tb refer to the global parameter order (as in params.dat), so the mask is built after the gather
    if isnothing(update_tb)
        update_tb = get_update_tb(conf, nparams)
    elseif length(update_tb) ≠ nparams
        allequal(update_tb) || error("update_tb has $(length(update_tb)) entries, but the model has $nparams parameters.")
        update_tb = fill(first(update_tb), nparams)
    end
    params_per_strc = [[findfirst(p->p==param, param_labels) for param in basis.parameters] for basis in bases]
    model = TBModel(hs, ones(nparams), param_labels, params_per_strc, update_tb)
    init_params!(model, conf, initas=initas, warn=rank==0)
    radial && add_radial_exponents!(model, last.(tensors), conf, initas=initas)
    write_summary && write_model_summary(model, strcs, bases, comm, conf, rank=rank, label=label)
    return model
end

get_params_for_strc(model::TBModel, index) = model.params[model.params_per_strc[index]]

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
    Hr = get_empty_complex_hamiltonians(size(h[1, 1], 1), size(h, 2), mode)
    Threads.@threads for R in axes(h, 2)
        for v in axes(h, 1)
            if Hr[R] isa SparseMatrixCSC
                @. Hr[R] += h[v, R] * V[v]
            else
                add_scaled!(Hr[R], h[v, R], V[v])
            end
        end
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

get_hr(model::TBModel, V, mode; apply_soc=false) = get_hr(model.hs, V, mode, apply_soc=apply_soc)
get_hr(model::TBModel, mode; apply_soc=false) = get_hr(model, model.params, mode, apply_soc=apply_soc)
function get_hr(model::TBModel, mode, index::Int64; apply_soc=false)
    params = get_params_for_strc(model, index)
    return get_hr(model.hs[index], params, mode, apply_soc=apply_soc)
end

"""
    update!(model::TBModel, opt, dV)

Updates the parameters of the given TB model `model` using the provided optimizer `opt` and the gradient `dV`.

# Arguments
- `model`: A `TBModel` object that contains the parameters to be updated.
- `opt`: An optimization algorithm or method used to update the model's parameters.
- `dV`: The gradient of the loss w.r.t. to the model parameters.
"""
function update!(model::TBModel, opt, dV)
    has_exponents(model) || return update!(opt, model.params, dV)
    n = length(model.params)
    update!(opt, model.params, dV[1:n])
    update!(opt, model.radial.c, dV[n+1:end])
    apply_radial_exponents!(model)
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
function get_model_gradient(h, dL_dHr::Vector{<:AbstractMatrix}; soc=false, logr=nothing)
    dV = zeros(size(h, 1)); dlog = zeros(size(h, 1))
    # Real part (projected to the spin-free basis with SOC) of dL_dHr, computed once per R instead of once per parameter
    G = [soc ? gradient_apply_spin_basis(real.(dL_dHr[R])) : real.(dL_dHr[R]) for R in eachindex(dL_dHr)]
    @tasks for v in axes(h, 1)
        for R in eachindex(G)
            dV[v] += sum_product(h[v, R], G[R])
            isnothing(logr) || (dlog[v] += sum_product(h[v, R], G[R], logr[v, R]))
        end
    end
    return isnothing(logr) ? dV : (dV, dlog)
end

"""
    sum_product(A, B)
    sum_product(A::SparseMatrixCSC, B, w)

Return `sum(A .* B)`, visiting only the nonzeros of `A` if it is sparse; with `w`, the nonzeros of `A` are weighted with
`w` (in the order of `nonzeros(A)`).
"""
sum_product(A::AbstractMatrix, B::AbstractMatrix) = sum(A .* B)
function sum_product(A::SparseMatrixCSC, B::AbstractMatrix, w=nothing)
    rows = rowvals(A); vals = nonzeros(A)
    s = zero(promote_type(eltype(A), eltype(B)))
    @inbounds for col in axes(A, 2), n in nzrange(A, col)
        s += (isnothing(w) ? vals[n] : vals[n] * w[n]) * B[rows[n], col]
    end
    return s
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
function get_model_gradient(model::TBModel, indices, reg, dL_dHr; soc=false)
    fit_exponents = has_exponents(model) && model.radial.update
    any(model.update) || fit_exponents || return zeros(length(get_params(model)))
    dV = zeros(length(model.params)); dlog = zeros(length(model.params))
    for (n, index) in enumerate(indices)
        p = model.params_per_strc[index]
        if fit_exponents
            g, glog = get_model_gradient(model.hs[index], dL_dHr[n], soc=soc, logr=model.radial.logr[index])
            dlog[p] .+= glog
        else
            g = get_model_gradient(model.hs[index], dL_dHr[n], soc=soc)
        end
        dV[p] .+= g
    end
    dV = @. ifelse(model.update, dV + $backward(reg, model.params), 0.)
    has_exponents(model) || return dV
    return vcat(dV, exponent_gradient(model, dlog))
end

"""
    exponent_gradient(model::TBModel, dlog)

Gradient of the loss w.r.t. the radial exponents from `dlog[v] = Σ G ⊙ h[v] ⊙ ln(d / Å)` (summed over the structures):
`∂H/∂c_v = -V_v h[v] ln(d / Å)`. Zero for exponents that are not fitted.
"""
function exponent_gradient(model::TBModel, dlog)
    dc = zeros(length(model.params))
    model.radial.update || return dc
    for v in hopping_parameters(model)
        dc[v] = -model.params[v] * dlog[v]
    end
    return dc
end

"""
    init_params!(model, basis, conf=get_empty_config(); initas=get_init_params(conf))

Initialize the parameters of a `model` based on the provided configuration and initialization method.

# Arguments
- `model`: The model whose parameter array `V` will be initialized.
- `conf`: Configuration settings that can be used to customize how parameters are initialized. By default, an empty configuration is used.
- `initas`: Initialization method (e.g., `ones`, `random` or a file).

# Keyword Arguments
- `conf`: Configuration settings.
- `initas`: The initialization method or file path for parameters (see [`get_init_mode`](@ref)). Defaults to `ones`.
- `warn`: Warn about model parameters that are missing in the parameter file (they keep the value 1).
"""
function init_params!(model, conf=get_empty_config(); initas=get_init_params(conf), warn=true)
    init_mode = get_init_mode(initas)
    if init_mode == :ones
        set_params!(model, ones(length(model.params)))
    elseif init_mode == :rand
        set_params!(model, rand(length(model.params)))
    elseif init_mode == :zeros
        set_params!(model, zeros(length(model.params)))
    else
        parameters, parameter_values, _, _, conf_values = read_params(initas)
        check_consistency(conf_values, conf)
        file_values = Dict(zip(parameters, parameter_values))
        missing_labels = similar(model.param_labels, 0)
        for (v, model_param) in enumerate(model.param_labels)
            if haskey(file_values, model_param)
                model.params[v] = file_values[model_param]
            else
                push!(missing_labels, model_param)
            end
        end
        if warn && !isempty(missing_labels)
            @warn "$(length(missing_labels)) of $(length(model.params)) TB parameters are not in $initas and start from 1: " *
                  join(string.(missing_labels[1:min(end, 10)]), ", ") * (length(missing_labels) > 10 ? ", ..." : "")
        end
    end
end

"""
    get_params(model::TBModel)

Retrieve the parameters associated with a `TBModel`.

# Arguments
- `model::TBModel`: The tight-binding model instance from which to extract parameters.

# Returns
- The parameters stored in the `params` field of the given `TBModel` instance, followed by the radial exponents if the
  model has them (`radial_exponent`).
"""
get_params(model::TBModel) = has_exponents(model) ? vcat(model.params, model.radial.c) : model.params

"""
    write_params(model::TBModel, conf=get_empty_config())

Writes the parameters of a `TBModel` using its parameter labels and values, and the radial exponents of the hopping
parameters as `exponent_<label>` lines.

# Arguments
- `model::TBModel`: The tight-binding model whose parameters need to be written.
- `conf`: (Optional) Configuration settings, defaults to an empty configuration.
"""
function write_params(model::TBModel, conf=get_empty_config())
    hoppings = has_exponents(model) ? hopping_parameters(model) : Int64[]
    write_params(model.param_labels, model.params, conf, exponent_labels=model.param_labels[hoppings],
                 exponents=has_exponents(model) ? model.radial.c[hoppings] : Float64[])
end

"""
    set_params!(model::TBModel, params)

Set the parameters of a `TBModel` instance while ensuring consistency with the model's structure.

# Arguments
- `model::TBModel`: The tight-binding model whose parameters are to be updated.
- `params`: The new parameter vector to assign to the model's `params` field. With radial exponents, either the
  parameters or the parameters followed by the exponents (as returned by `get_params`).

# Error Conditions
- Throws an error if the parameter vector `params` is not of the correct size.

# Returns
- Updates the `params` field of the `model` in place if the consistency checks pass.
"""
function set_params!(model::TBModel, params)
    n = length(model.params)
    if has_exponents(model) && length(params) == 2n
        model.params .= params[1:n]
        model.radial.c .= params[n+1:end]
        apply_radial_exponents!(model)
    elseif size(model.params) ≠ size(params)
        error("Parameter vector is not of correct size!")
    else
        model.params = params
    end
end

"""
    copy_params!(receiving_model::TBModel, sending_model::TBModel) -> Nothing

Copy matching parameters (and radial exponents, if both models have them) from one TB model to another.

# Arguments
- `receiving_model::TBModel`: The model whose parameters will be updated.
- `sending_model::TBModel`: The model providing parameter values.
"""
function copy_params!(receiving_model::TBModel, sending_model::TBModel)
    exponents = has_exponents(receiving_model) && has_exponents(sending_model)
    for (i, sending_label) in enumerate(sending_model.param_labels)
        for (j, receiving_label) in enumerate(receiving_model.param_labels)
            if sending_label == receiving_label
                receiving_model.params[j] = sending_model.params[i]
                exponents && (receiving_model.radial.c[j] = sending_model.radial.c[i])
            end
        end
    end
    exponents && apply_radial_exponents!(receiving_model)
end
