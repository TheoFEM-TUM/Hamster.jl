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
function get_eigenvalue_gradient(vs, Rs, ks)
    Nε = size(vs, 1); NR = size(Rs, 2); Nk = size(ks, 2)
    dE_dHr = Array{Matrix{Float64}}(undef, NR, Nε, Nk)
    hellman_feynman!(dE_dHr, vs, exp_2πi(Rs, ks))
    return dE_dHr
end

function get_eigenvalue_gradient(vs::AbstractArray{<:SparseVector}, Rs, ks)
    Nε = size(vs, 1); NR = size(Rs, 2); Nk = size(ks, 2)
    dE_dHr = Array{SparseMatrixCSC{Float64, Int64}}(undef, NR, Nε, Nk)
    hellman_feynman!(dE_dHr, vs, exp_2πi(Rs, ks))
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
function hellman_feynman!(dE_dλ, Ψ_i, dHk_dHr)
    for k in axes(Ψ_i, 2)
        for m in axes(Ψ_i, 1)
            for R in axes(dHk_dHr, 1)
                @views dE_dλ[R, m, k] = @. real(conj(Ψ_i[m, k])' * dHk_dHr[R, k] * Ψ_i[m, k])
            end
        end
    end
end