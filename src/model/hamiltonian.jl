"""
    get_exp_ikR(k⃗, R⃗)
    
Calculate the phase factor exp(2πik⃗⋅R⃗).
"""
@inline exp_2πi(R⃗, k⃗) = @. exp(2π*im * $*(R⃗', k⃗))

get_empty_hamiltonians(Nε, NkR; sp_mode=false) = [ifelse(sp_mode, spzeros, zeros)(ComplexF64, Nε, Nε) for _ in 1:NkR]

"""
    get_hamiltonian(Hᴿ, Rs, ks)

Calculate the Hamiltonian for the real-space Hamiltonian matrices `Hᴿ` at grid
points `Rs` and the k-points `ks`.
"""
function get_hamiltonian(Hᴿ::Array{Float64, 3}, Rs, ks)
    Nε = size(Hᴿ, 1); Nk = size(ks, 2)
    Hᵏ = zeros(ComplexF64, Nε, Nε, Nk); Hᴿ = complex.(Hᴿ)
    exp_2πiRk = exp_2πi(Rs, ks)
    @tensor Hᵏ[i, j, k] = Hᴿ[i, j, R] * exp_2πiRk[R, k]
    return Hᵏ
end

"""
    get_hamiltonian(Hr::Vector{<:AbstractMatrix}, Rs, ks; sp_mode=false)

Constructs the Hamiltonian matrices `Hk` in k-space by summing over real-space Hamiltonians `Hr` weighted by phase factors derived from `Rs` and `ks`.

# Arguments:
- `Hr::Vector{<:AbstractMatrix}`: A vector of real-space Hamiltonian matrices. Each element in `Hr` corresponds to a Hamiltonian for a specific lattice vector `Rs`.
- `Rs`: A collection of lattice vectors corresponding to the Hamiltonians in `Hr`. Typically, this is a matrix where each row is a lattice vector.
- `ks`: A collection of momentum vectors `k` for which the Hamiltonian matrices `Hk` are to be computed. Typically, this is a matrix where each column is a momentum vector `k`.
- `sp_mode::Bool=false`: A boolean flag indicating whether to use sparse matrices.

# Returns:
- `Hk`: A vector of Hamiltonian matrices in momentum space. Each `Hk[k]` corresponds to the Hamiltonian for a specific k-vector `k`.
"""
function get_hamiltonian(Hr::Vector{<:AbstractMatrix}, Rs, ks; sp_mode=false)
    Hk = get_empty_hamiltonians(size(Hr[1], 1), size(ks, 2); sp_mode=sp_mode)
    exp_2πiRk = exp_2πi(Rs, ks)
    Threads.@threads for k in eachindex(Hk)
        @views Hk[k] = sum(Hr .* exp_2πiRk[:, k])
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
- `eigenvectors::Matrix{Float64}`: A matrix where each column is an eigenvector corresponding to one of the computed eigenvalues.
"""
function diagonalize(Hk::SparseMatrixCSC; Neig=6, target=0)
    Es, vs = eigsolve(Hk, Neig, EigSorter(λ->abs(target-λ), rev=false), ishermitian=true)
    if abs(Es[1] - target) > abs(Es[end] - target)
        return real.(Es[end-Neig:end]), hcat(vs[end-Neig:end]...)
    else
        return real.(Es[1:Neig]), hcat(vs[1:Neig]...)
    end
end