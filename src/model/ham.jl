"""
    exp_2πi(k⃗, R⃗)
    
Calculate the phase factor exp(2πik⃗⋅R⃗).
"""
@inline exp_2πi(R⃗, k⃗) = @. exp(2π*im * $*(R⃗', k⃗))

get_empty_complex_hamiltonians(Nε, NkR, mode=Dense()) = Matrix{ComplexF64}[zeros(ComplexF64, Nε, Nε) for _ in 1:NkR]
get_empty_complex_hamiltonians(Nε, NkR, ::Sparse) = SparseMatrixCSC{ComplexF64, Int64}[spzeros(ComplexF64, Nε, Nε) for _ in 1:NkR]
get_empty_real_hamiltonians(Nε, NkR, mode=Dense()) = Matrix{Float64}[zeros(Float64, Nε, Nε) for _ in 1:NkR]
get_empty_real_hamiltonians(Nε, NkR, ::Sparse) = SparseMatrixCSC{Float64, Int64}[spzeros(Float64, Nε, Nε) for _ in 1:NkR]

"""
    get_hamiltonian(Hr::Vector{<:AbstractMatrix}, Rs, ks, mode=Dense(), weights=ones(size(Rs, 2)))

Constructs a vector of Hamiltonian matrices by combining a vector of matrices `Hr` with phase factors determined by `Rs` and `ks`.

# Arguments:
- `Hr::Vector{<:AbstractMatrix}`: A vector of matrices where each matrix represents a component of the Hamiltonian. These matrices must be compatible in size for the operations performed.
- `Rs`: A matrix or array containing positional information used to calculate phase factors.
- `ks`: A matrix of momentum values used in conjunction with `Rs` to compute phase factors.
- `mode::Dense()`: Indicates whether to construct a sparse or dense Hamiltonian. Defaults to `Dense`.
- `weights::Vector=ones(size(Rs, 2))`: A vector of weights used in the summation of phase factors. The length of this vector should match the number of columns in `Rs`. These are the degeneracies for the Wannier90 Hamiltonians.

# Returns:
- A vector of Hamiltonian matrices `Hk`, where each matrix is constructed by combining `Hr` with phase factors and optionally transformed into a spin basis.
"""
function get_hamiltonian(Hr::Vector{<:AbstractMatrix}, Rs, ks, mode=Dense(); weights=ones(size(Rs, 2)))
    Nε = size(Hr[1], 1)
    Hk = get_empty_complex_hamiltonians(Nε, size(ks, 2), mode)
    exp_2πiRk = exp_2πi(Rs, ks)

    @tasks for k in eachindex(Hk)
        @views @inbounds for R in eachindex(Hr)
            @. Hk[k] += Hr[R] * exp_2πiRk[R, k] * weights[R]
        end
    end
    return Hk
end

"""
    diagonalize(Hk::Vector{<:AbstractMatrix}; Neig=size(Hk[1], 1), target=0)

Diagonalizes a vector of Hamiltonian matrices `Hk` and returns the specified number of eigenvalues and eigenvectors for each matrix.

# Arguments:
- `Hk::Vector{<:AbstractMatrix}`: A vector of Hamiltonian matrices to be diagonalized. Each matrix typically corresponds to a different momentum `k` in a band structure calculation.
- `Neig::Int=size(Hk[1], 1)`: The number of eigenvalues and corresponding eigenvectors to compute for each Hamiltonian matrix. Defaults to the full diagonalization (`size(Hk[1], 1)`).
- `target::Real=0`: The target eigenvalue around which to focus the computation. This is useful when using methods like the Lanczos algorithm to compute eigenvalues near a specific energy.

# Returns:
- `Es::Matrix{Float64}`: A matrix where each column `Es[:, k]` contains the `Neig` eigenvalues of the `k`-th Hamiltonian matrix in `Hk`.
- `vs::Array{ComplexF64, 3}`: A 3D array where each `vs[:, :, k]` contains the `Neig` eigenvectors corresponding to the `k`-th Hamiltonian matrix in `Hk`. The dimensions of `vs` are `(Nε, Neig, Nk)`, where `Nε` is the size of each Hamiltonian matrix, `Neig` is the number of eigenvectors, and `Nk` is the number of Hamiltonian matrices.
"""
function diagonalize(Hk::Vector{<:AbstractMatrix}; Neig=size(Hk[1], 1), target=0)
    Nε = size(Hk[1], 1)
    Nk = length(Hk)
    Es = zeros(Neig, Nk)
    vs = zeros(ComplexF64, Nε, Neig, Nk)

    Threads.@threads for k in eachindex(Hk)
        Es[:, k], vs[:, :, k] = diagonalize(Hk[k], Neig=Neig, target=target)
    end

    return Es, vs
end

"""
    diagonalize(Hk::AbstractMatrix; Neig=size(Hk, 1), target=0)

Fully diagonalizes a Hermitian Hamiltonian matrix `Hk` and returns the eigenvalues and eigenvectors.

# Arguments:
- `Hk::AbstractMatrix`: A Hermitian matrix (Hamiltonian) to be diagonalized. The matrix should be square and typically complex-valued.
- `Neig::Int=size(Hk, 1)`: The number of eigenvalues and corresponding eigenvectors to compute. Not used.
- `target::Real=0`: The target eigenvalue around which to focus the computation. Not used.

# Returns:
- `real_values::Vector{Float64}`: A vector containing the real parts of the eigenvalues of `Hk`. The eigenvalues are computed using the Hermitian matrix, so they are guaranteed to be real.
- `eigenvectors::Matrix{ComplexF64}`: A matrix where each column is an eigenvector corresponding to an eigenvalue of `Hk`. The eigenvectors are computed in the standard basis and are complex-valued.
"""
function diagonalize(Hk::AbstractMatrix; Neig=size(Hk, 1), target=0)
    eig = eigen(Hermitian(Hk))
    return real.(eig.values), eig.vectors
end

"""
    diagonalize(Hk::SparseMatrixCSC; Neig=6, target=0)

Diagonalizes a sparse Hermitian matrix `Hk` to find a specified number of eigenvalues and eigenvectors, optionally focusing on eigenvalues near a given target.

# Arguments:
- `Hk::SparseMatrixCSC`: A sparse Hermitian matrix in compressed sparse column format to be diagonalized. The matrix should be square and Hermitian.
- `Neig::Int=6`: The number of eigenvalues and corresponding eigenvectors to compute. Defaults to `6`, but can be adjusted based on the required precision or size of the spectrum.
- `target::Real=0`: The target eigenvalue around which to focus the computation. This is used to prioritize finding eigenvalues closest to this value. Defaults to `0`.

# Returns:
- `eigenvalues::Vector{Float64}`: A vector of the real parts of the computed eigenvalues, focusing on those closest to the target. The number of eigenvalues returned is equal to `Neig`.
- `eigenvectors::Matrix{ComplexF64}`: A matrix where each column is an eigenvector corresponding to one of the computed eigenvalues.
"""
function diagonalize(Hk::SparseMatrixCSC; Neig=6, target=0)
    Es, vs = eigsolve(Hk, Neig, EigSorter(λ->abs(target-λ), rev=false), ishermitian=true)
    if abs(Es[1] - target) > abs(Es[end] - target)
        return real.(Es[end-Neig:end]), hcat(vs[end-Neig:end]...)
    else
        return real.(Es[1:Neig]), hcat(vs[1:Neig]...)
    end
end

"""
    get_sparsity(H::AbstractArray; sp_tol=1e-10)

Calculates the sparsity of a matrix or array by determining the fraction of elements that are considered effectively zero based on a specified numerical tolerance.

# Arguments:
- `H::AbstractArray`: The input matrix or array for which sparsity is to be calculated. It can be a dense or sparse matrix, or any array-like structure.
- `sp_tol::Real=1e-10`: The numerical tolerance used to determine whether an element is considered zero. Elements with both real and imaginary parts less than `sp_tol` in magnitude are considered zero.

# Returns:
- `sparsity::Float64`: The fraction of elements in `H` that are considered zero according to the specified tolerance. This value lies between `0.0` (no zero elements) and `1.0` (all elements are effectively zero).
"""
function get_sparsity(H::AbstractArray; sp_tol=1e-10)
    Ntot = prod(size(H))
    Nzero = 0
    for i in eachindex(H)
        if (abs(real(H[i])) < sp_tol) && (abs(imag(H[i])) < sp_tol)
            Nzero += 1
        end
    end
    return Nzero / Ntot
end

"""
    get_sparsity(H::Vector{AbstractArray}; sp_tol=1e-10)

Calculates the sparsity of a vector of arrays by determining the fraction of elements across all arrays that are considered effectively zero based on a specified numerical tolerance.

# Arguments:
- `H::Vector{AbstractArray}`: A vector containing arrays (e.g., matrices) for which sparsity is to be calculated. Each element of `H` should be an array of the same size.
- `sp_tol::Real=1e-10`: The numerical tolerance used to determine whether an element is considered zero. Elements with both real and imaginary parts less than `sp_tol` in magnitude are considered zero.

# Returns:
- `sparsity::Float64`: The fraction of elements across all arrays in `H` that are considered zero according to the specified tolerance. This value lies between `0.0` (no zero elements) and `1.0` (all elements are effectively zero).
"""
function get_sparsity(H::Vector{<:AbstractArray}; sp_tol=1e-10)
    Ntot = prod(size(H[1])) * length(H)
    Nzero = 0
    for k in eachindex(H), inds in eachindex(H[k])
        if (abs(real(H[k][inds])) < sp_tol) && (abs(imag(H[k][inds])) < sp_tol)
            Nzero += 1
        end
    end
    return Nzero / Ntot
end

"""
    droptol!(H::Vector{AbstractSparseMatrix}, tol=1e-8)

Applies a tolerance to drop small elements from a vector of sparse matrices, modifying the matrices in place.

# Arguments:
- `H::Vector{AbstractSparseMatrix}`: A vector of sparse matrices where small elements will be dropped. The matrices are modified in place.
- `tol::Real=1e-8`: The numerical tolerance used to determine which elements are considered too small and should be dropped. Elements with absolute values less than `tol` are removed from the sparse matrices.
"""
function SparseArrays.droptol!(H::Vector{<:AbstractSparseMatrix}, tol=1e-10)
    for k in eachindex(H)
        droptol!(H[k], tol)
    end
end

"""
    apply_spin_basis(H::AbstractMatrix; alternating_order=false)

Extends a given matrix `H` to a spin basis (up/down) representation by applying the tensor product with the identity matrix. The order of the tensor product application is controlled by the `alternating_order` flag which affects the order of spin states in the basis.

# Arguments:
- `H::AbstractMatrix`: The input matrix to be extended to the spin basis. It should be a square matrix or generally a 2D array.
- `alternating_order::Bool=false`: A boolean flag that determines the order of applying the spin basis. If `true`, the order is, e.g., up,down,up,down. If `false`, the order is, e.g., up,up,down,down.

# Returns:
- A matrix in the spin basis in the specified order.
"""
function apply_spin_basis(H::AbstractMatrix; alternating_order=false)
    I_spin = Array{Int64}(I, 2, 2)
    return alternating_order ? kron(I_spin, H) : kron(H, I_spin)
end

"""
    gradient_apply_spin_basis(dHr::AbstractMatrix; alternating_order=false)

Calculate the gradient of the spin basis applied to a given matrix `dHr`. The output matrix is of size `(Nε/2, Nε/2)`.

This function reshapes the input matrix `dHr`, which is expected to represent a gradient in a spin system, into a form suitable for applying the spin basis transformation. The transformation is performed using the Kronecker product, and the output is computed depending on the specified order of application.

# Arguments
- `dHr::AbstractMatrix`: An abstract matrix containing the gradient data to be transformed. It should have a shape that is compatible with the spin basis operations.
- `alternating_order::Bool`: A flag that determines the order of the spin basis application. If `false`, the function applies the transformation as `kron(H, I_spin)`. If `true`, it applies the transformation as `kron(I_spin, H)`.
"""
function gradient_apply_spin_basis(dHr::AbstractMatrix; alternating_order=false)
    Nε = size(dHr, 1) ÷ 2
    if !alternating_order
        # Case 1: kron(H, I_spin)
        dHr_out = sum(reshape(dHr, 2, Nε, 2, Nε), dims=(1, 3))
        return dropdims(dHr_out, dims=(1, 3))
    else
        # Case 2: kron(I_spin, H)
        dHr_out = sum(reshape(dHr, Nε, 2, Nε, 2), dims=(2, 4))
        return dropdims(dHr_out, dims=(2, 4))
    end
end

"""
    reshape_and_sparsify_eigenvectors(vs, mode::SparsityMode; sp_tol=1e-10) -> Matrix

Reshapes and optionally sparsifies a 3D array of eigenvectors `vs` into a 2D matrix of vectors, 
depending on the specified `SparsityMode`. 

The input `vs` is assumed to have dimensions `(n, m, k)`, where:
- `n` represents the size of each eigenvector.
- `m` and `k` represent the number of eigenvector groups along two axes.

# Arguments
- `vs::Array`: A 3D array of eigenvectors, where `vs[:, m, k]` corresponds to the eigenvector at position `(m, k)`.
- `mode::SparsityMode`: Specifies the sparsity mode. Must be either:
  - `Dense`: Produces a dense matrix where each element is a dense vector.
  - `Sparse`: Produces a sparse matrix where each element is a sparse vector.
- `sp_tol::Float64=1e-10`: (Optional) The tolerance below which elements of the eigenvectors are dropped when 
  `Sparse` mode is selected. Defaults to `1e-10`.

# Returns
- `Matrix{Vector{ComplexF64}}` if `mode` is `Dense`: A 2D matrix of dense vectors corresponding to eigenvectors in `vs`.
- `Matrix{SparseVector{ComplexF64, Int64}}` if `mode` is `Sparse`: A 2D matrix of sparse vectors, where elements smaller 
  than `sp_tol` are removed.
"""
function reshape_and_sparsify_eigenvectors(vs, ::Dense; sp_tol=1e-10)
    vs_out = Matrix{Vector{ComplexF64}}(undef, size(vs, 2), size(vs, 3))
    for k in axes(vs, 3)
        for m in axes(vs, 2)
            @views vs_out[m, k] = vs[:, m, k]
        end
    end
    return vs_out
end

function reshape_and_sparsify_eigenvectors(vs, ::Sparse; sp_tol=1e-10)
    vs_out = Matrix{SparseVector{ComplexF64, Int64}}(undef, size(vs, 2), size(vs, 3))
    for k in axes(vs, 3)
        for m in axes(vs, 2)
            sparse_vec = sparse(vs[:, m, k])
            droptol!(sparse_vec, sp_tol)
            @views vs_out[m, k] = sparse_vec
        end
    end
    return vs_out
end