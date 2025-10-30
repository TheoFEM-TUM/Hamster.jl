"""
    get_eigenvalue_gradient(vs, Rs, ks, sp_mode, sp_iterator; nthreads_bands, nthreads_kpoints)

Compute the gradient `dE_dHr` of each energy eigenvalue at each k-point w.r.t. the real-space Hamiltonian matrix elements.

# Arguments
-`vs`: Matrix that contains the (sparse) eigenvectors of each eigenvalue `(m, k)`.
- `Rs::Matrix{Int64}`: Lattice translation vectors in units of the lattice vectors.
- `ks::Matrix{Float64}`: The coordinates of each k-point in units of the reciprocal lattice vectors.
- `sp_mode`: Either `Dense` or `Sparse`, determines whether the gradients are sparsified.
- `sp_iterator`: Special iterator to optimize iteration over `i, j, R` for very sparse systems.

# Returns
- `dE_dHr`: An array of shape `(NR, Nε, Nk)` that contains (sparse) matrices of shape `(Nε, Nε)`.
"""
function get_eigenvalue_gradient(vs, Rs, ks, ::Dense, sp_iterator=nothing; nthreads_bands=Threads.nthreads(), nthreads_kpoints=Threads.nthreads(), sp_tol=1e-10)
    Nε = size(vs, 1); NR = size(Rs, 2); Nk = size(ks, 2)
    dE_dHr = Array{Matrix{ComplexF64}}(undef, NR, Nε, Nk)
    tforeach(axes(vs, 3), nchunks=nthreads_kpoints) do k
        tforeach(axes(vs, 2), nchunks=nthreads_bands) do m
            @views for R in 1:NR
                dE_dHr[R, m, k] = zeros(Nε, Nε)
            end
        end
    end
    hellman_feynman!(dE_dHr, vs, exp_2πi(Rs, ks), nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands)
    return dE_dHr
end

function get_eigenvalue_gradient(vs, Rs, ks, ::Sparse, sp_iterator; nthreads_bands=Threads.nthreads(), nthreads_kpoints=Threads.nthreads(), sp_tol=1e-10)
    Nε = size(vs, 1); NR = size(Rs, 2); Nk = size(ks, 2)
    dE_dHr = fill(spzeros(ComplexF64, Nε, Nε), NR, Nε, Nk)
    sparse_hellman_feynman!(dE_dHr, vs, exp_2πi(Rs, ks), sp_iterator, nthreads_kpoints=nthreads_kpoints, nthreads_bands=nthreads_bands, sp_tol=sp_tol)
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
function hellman_feynman!(dE_dHr, Ψ, dHk_dHr; nthreads_bands=Threads.nthreads(), nthreads_kpoints=Threads.nthreads())
    tforeach(axes(Ψ, 3), nchunks=nthreads_kpoints) do k
        tforeach(axes(Ψ, 2), nchunks=nthreads_bands) do m
            @views for R in axes(dHk_dHr, 1), j in axes(Ψ, 1), i in axes(Ψ, 1)
                dE_dHr[R, m, k][i, j] = conj(Ψ[i, m, k]) * dHk_dHr[R, k] * Ψ[j, m, k]
            end
        end
    end
end

function sparse_hellman_feynman!(dE_dHr, Ψ, dHk_dHr, sp_iterator; nthreads_kpoints=Threads.nthreads(), nthreads_bands=Threads.nthreads(), sp_tol=1e-10)
    max_nnz = maximum([length(inds) for inds in sp_iterator])
    max_nt = nthreads_bands + nthreads_kpoints
    Nε = size(Ψ, 1)
    
    # Preallocate thread-local buffers
    thread_buffers = Dict(tid => (
        is = Vector{Int64}(undef, max_nnz),
        js = Vector{Int64}(undef, max_nnz),
        vals = Vector{ComplexF64}(undef, max_nnz)
    ) for tid in 1:max_nt)

    tforeach(axes(Ψ, 3), nchunks=nthreads_kpoints) do k
        tforeach(axes(Ψ, 2), nchunks=nthreads_bands) do m
            id = Threads.threadid()
            buffer = thread_buffers[id]
            @unpack is, js, vals = buffer
            @views for (R, inds) in enumerate(sp_iterator)
                nnz = 0
                for (i, j) in inds
                    val = conj(Ψ[i, m, k]) * dHk_dHr[R, k] * Ψ[j, m, k]
                    if abs(val) > sp_tol
                        nnz += 1
                        is[nnz] = i; js[nnz] = j; vals[nnz] = val
                    end
                end
                dE_dHr[R, m, k] = sparse(is[1:nnz], js[1:nnz], vals[1:nnz], Nε, Nε)
            end
        end
    end
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
    max_nt = nthreads_bands + nthreads_kpoints
    dL_dHr_thread = [get_empty_complex_hamiltonians(size(dE_dHr, 2), size(dE_dHr, 1), mode) for _ in 1:max_nt]
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