"""
    mutable struct HamiltonianKernel{T1, T2}

A kernel structure used for computing weighted similarity functions.

# Fields
- `ws :: Vector{Float64}`: Weights for each sample point.
- `xs :: Vector{T1}`: Sample points.
- `sim_params :: T2`: Parameters for the similarity function.
"""
mutable struct HamiltonianKernel{T1, T2, T3}
    params :: Vector{Float64}
    data_points :: Vector{T1}
    sim_params :: T2
    structure_descriptors :: Vector{T3}
end

"""
    HamiltonianKernel(strcs, bases, model, conf)

Constructor for a HamiltonianKernel model.
"""
function HamiltonianKernel(strcs, bases, model, comm, conf=get_empty_config(); Ncluster=get_ml_ncluster(conf), Npoints=get_ml_npoints(conf), sim_params=get_sim_params(conf), rank=0, nranks=1)
    structure_descriptors = map(eachindex(strcs)) do n
        get_tb_descriptor(model.hs[n], model.V, strcs[n], bases[n], conf)
    end
    Npoints_local = floor(Int64, Npoints / nranks)
    data_points_local = sample_structure_descriptors(reshape_structure_descriptors(structure_descriptors), Ncluster=Ncluster, Npoints=Npoints_local)

    data_points = MPI.Reduce(data_points_local, vcat, comm, root=0)
    MPI.Bcast!(data_points, comm, root=0)

    # TODO: init params
    params = zeros(length(data_points))
    return HamiltonianKernel(params, data_points, sim_params, structure_descriptors)
end

exp_sim(x₁, x₂; σ=0.1)::Float64 = exp(-normdiff(x₁, x₂)^2 / σ)

(k::HamiltonianKernel)(xin) = mapreduce(wx->wx[1]*exp_sim(wx[2], xin, σ=k.sim_params), +, zip(k.params, k.data_points))

"""
    get_hr(kernel::HamiltonianKernel, mode, index; apply_soc=false) -> Vector{Matrix{Float64}}

Constructs a set of real-space Hamiltonians from a `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel used for computing matrix elements.
- `mode`: Specifies the sparsity mode.
- `index`: Index specifying which structure to evaluate.

# Keyword Arguments
- `apply_soc`: If `true`, applies the spin-orbit coupling (SOC) basis transformation.

# Returns
- A vector of real-space Hamiltonian matrices, optionally modified with SOC transformations.
"""
function get_hr(kernel::HamiltonianKernel, mode::Dense, index; apply_soc=false)
    h_env = kernel.structure_descriptors[index]
    Hr = get_empty_real_hamiltonians(size(h_env[1], 1), length(h_env), mode)
    for R in eachindex(h_env)
        for (i, j, hin) in zip(findnz(h_env[R])...)
            Hr[R][i, j] = kernel(hin)
        end
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

function get_hr(kernel::HamiltonianKernel, mode::Sparse, index; apply_soc=false)
    h_env = kernel.structure_descriptors[index]
    Nε = size(h_env[1], 1)
    Hr = get_empty_real_hamiltonians(Nε, length(h_env), mode)
    for R in eachindex(h_env)
        is = Int64[]
        js = Int64[]
        vals = Float64[]
        for (i, j, hin) in zip(findnz(h_env[R])...)
            push!(is, i); push!(js, j); push!(vals, kernel(hin))
        end
        Hr[R] = sparse(is, js, vals, Nε, Nε)
    end
    return apply_soc ? apply_spin_basis.(Hr) : Hr
end

"""
    update!(kernel::HamiltonianKernel, opt, grad)

Updates the parameters of a `HamiltonianKernel` using an optimization method `opt`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel whose parameters are to be updated.
- `opt`: The optimizer used to perform the update.
- `grad`: The gradient used for updating the parameters.
"""
function update!(kernel::HamiltonianKernel, opt, grad)
    update!(opt, kernel.params, grad)
end

"""
    get_params(kernel::HamiltonianKernel)

Retrieve the parameters associated with a `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel instance from which to extract parameters.

# Returns
- The parameters stored in the `ws` field of the given `HamiltonianKernel` instance.
"""
get_params(kernel::HamiltonianKernel) = kernel.params

function write_params(kernel::HamiltonianKernel, conf=get_empty_conf())
    for n in eachindex(kernel.params)
        
    end
end

"""
    set_params!(kernel::HamiltonianKernel, ws)

Set the parameters of a `HamiltonianKernel` instance.

# Arguments
- `kernel::HamiltonianKernel`: The kernel model whose parameters are to be updated.
- `ws`: The new parameter vector.

# Error Conditions
- Throws an error if the parameter vector `ws` is not of the correct size.

# Returns
- Updates the `Vs` field of the `kernel` in place if the consistency checks pass.
"""
function set_params!(kernel::HamiltonianKernel, params)
    throw_error = size(kernel.params) ≠ size(params)
    if throw_error
        error("Parameter vector is not of correct size!")
    else
        kernel.params = params
    end
end

"""
    get_model_gradient(kernel::HamiltonianKernel, indices, reg, dL_dHr) -> Vector{Float64}

Computes the gradient of the model parameters for a given `HamiltonianKernel`.

# Arguments
- `kernel::HamiltonianKernel`: The Hamiltonian kernel for which the gradient is computed.
- `indices`: Indices specifying which structure descriptors to use.
- `reg`: Regularization term.
- `dL_dHr`: Gradient of the loss function with respect to the real-space Hamiltonian.

# Returns
- `dparams`: A vector containing the computed gradients of the model parameters.
"""
function get_model_gradient(kernel::HamiltonianKernel, indices, reg, dL_dHr)
    dparams = zeros(length(kernel.params))
    @views for n in eachindex(dparams), index in indices
        h_env = kernel.structure_descriptors[index]
        for R in eachindex(dL_dHr[index]), (i, j, hin) in zip(findnz(h_env[R])...)
            dparams[n] += exp_sim(kernel.data_points[n], hin, σ=kernel.sim_params) * dL_dHr[index][R][i, j]
        end
    end
    dparams_penal = backward(reg, kernel.params)
    return dparams .+ dparams_penal
end