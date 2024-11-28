"""
    get_eigenvalue_gradient(vs, Rs, ks)
    get_eigenvalue_gradient(vs::AbstractArray{<:SparseVector}, Rs, ks)

Compute the gradient `dE_dHr` of each energy eigenvalue at each k-point w.r.t. the real-space Hamiltonian matrix elements.

# Arguments
-`vs`: Matrix that contains the (sparse) eigenvectors of each eigenvalue `(m, k)`.
- `Rs::Matrix{Int64}`: Lattice translation vectors in units of the lattice vectors.
- `ks::Matrix{Float64}`: The coordinates of each k-point in units of the reciprocal lattice vectors.

# Returns
- `dE_dHr`: An array of shape `(NR, Nε, Nk)` that contains (sparse) matrices of shape `(Nε, Nε)`.
"""
function get_eigenvalue_gradient(vs, Rs, ks; nthreads_bands=Threads.nthreads(), nthreads_kpoints=Threads.nthreads(), sp_tol=1e-10)
    Nε = size(vs, 1); NR = size(Rs, 2); Nk = size(ks, 2)
    dE_dHr = Array{Matrix{Float64}}(undef, NR, Nε, Nk)
    hellman_feynman!(dE_dHr, vs, exp_2πi(Rs, ks), nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands, sp_tol=sp_tol)
    return dE_dHr
end

function get_eigenvalue_gradient(vs::AbstractArray{<:SparseVector}, Rs, ks; nthreads_bands=Threads.nthreads(), nthreads_kpoints=Threads.nthreads(), sp_tol=1e-10)
    Nε = size(vs, 1); NR = size(Rs, 2); Nk = size(ks, 2)
    dE_dHr = Array{SparseMatrixCSC{Float64, Int64}}(undef, NR, Nε, Nk)
    hellman_feynman!(dE_dHr, vs, exp_2πi(Rs, ks), nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands, sp_tol=sp_tol)
    return dE_dHr
end

"""
    hellman_feynman!(dE_dλ, Ψ_i, dHk_ij)

Compute the gradient of the eigenvalues using the Hellmann-Feynman theorem and store the results in `dE_dλ`.

# Arguments
- `dE_dλ::AbstractArray`: Pre-allocated array to store the computed gradients (as Nε×Nε matrix), typically of size `(NR, Nε, Nk)`. 
- `Ψ_i::AbstractArray`: Array of eigenvectors (wavefunctions), typically of size `(Nε, Nk)`.
- `dHk_ij::AbstractArray`: Array representing the derivative of the Hamiltonian with respect some `λ` for each lattice vector and k-point. Its size is typically `(NR, Nk)`.

# Details
For each k-point `k` and eigenstate `m`, this function computes the matrix element of the form:

    ⟨Ψ_i[m, k] | dHk_ij[R, k] | Ψ_i[m, k]⟩

and stores the real part of this value in `dE_dλ[R, m, k]`. This is done for each lattice vector `R`, eigenstate `m`, and k-point `k`.
"""
function hellman_feynman!(dE_dHr, Ψ, dHk_dHr; nthreads_bands=Threads.nthreads(), nthreads_kpoints=Threads.nthreads(), sp_tol=1e-10)
    tforeach(axes(Ψ, 2), nchunks=nthreads_kpoints) do k
        tforeach(axes(Ψ, 1), nchunks=nthreads_bands) do m
            for R in axes(dHk_dHr, 1)
                @views dE_dHr[R, m, k] = _hellman_feynman_step(Ψ[m, k], dHk_dHr[R, k], sp_tol=sp_tol)
            end
        end
    end
end

function _hellman_feynman_step(Ψ_mk::AbstractVector, dHk_dHr; sp_tol=1e-10)
    dE_dHr = zeros(length(Ψ_mk), length(Ψ_mk))
    for i in eachindex(Ψ_mk), j in eachindex(Ψ_mk)
        dE_dHr[i, j] = real(conj(Ψ_mk[i]) * dHk_dHr * Ψ_mk[j])
    end
    return dE_dHr
end

function _hellman_feynman_step(Ψ_mk::SparseVector, dHk_dHr; sp_tol=1e-10)
    dE_dHr = spzeros(length(Ψ_mk), length(Ψ_mk))
    for i in nzrange(Ψ_mk, 1), j in nzrange(Ψ_mk, 1)
        dE_dHr[i, j] = real(conj(Ψ_mk[i]) * dHk_dHr * Ψ_mk[j])
    end
    droptol!(dE_dHr, sp_tol)
    return dE_dHr
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
function chain_rule(dL_dE, dE_dHr, mode; nthreads_bands=Threads.nthreads(), nthreads_kpoints=Threads.nthreads(), sp_tol=1e-10)
    dL_dHr_thread = [get_empty_real_hamiltonians(size(dE_dHr, 2), size(dE_dHr, 1), mode) for _ in 1:Threads.nthreads()]
    tforeach(axes(dE_dHr, 3), nchunks=nthreads_kpoints) do k
        tforeach(axes(dE_dHr, 2), nchunks=nthreads_bands) do m
            for R in axes(dE_dHr, 1)
                @views @. dL_dHr_thread[Threads.threadid()][R] += dL_dE[m , k] * dE_dHr[R, m, k]
            end
        end
    end
    dL_dHr = sum(dL_dHr_thread)
    if mode isa Sparse; droptol!(dL_dHr, sp_tol); end
    return dL_dHr
end