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
        @inbounds for R in eachindex(Hr)
            add_scaled!(Hk[k], Hr[R], exp_2πiRk[R, k] * weights[R])
        end
    end
    return Hk
end

function get_hamiltonian(Hr::Vector{<:AbstractMatrix}, Rs, ks, ::Sparse; weights=ones(size(Rs, 2)))
    Nε = size(Hr[1], 1)
    exp_2πiRk = exp_2πi(Rs, ks)
    # Nonzeros of all Hr[R], collected once; Hk[k] is then assembled in a single step per k-point
    Is, Js, Vs, Ridx = coo_entries(Hr)
    Hk = Vector{SparseMatrixCSC{ComplexF64, Int64}}(undef, size(ks, 2))
    @tasks for k in eachindex(Hk)
        vals = [Vs[n] * exp_2πiRk[Ridx[n], k] * weights[Ridx[n]] for n in eachindex(Vs)]
        Hk[k] = sparse(Is, Js, vals, Nε, Nε)
    end
    return Hk
end

"""
    add_scaled!(H::Matrix, A, c)

Add `c * A` to the dense matrix `H` in place. For sparse `A`, only its nonzeros are visited.
"""
add_scaled!(H::Matrix, A::AbstractMatrix, c) = (H .+= c .* A; H)
function add_scaled!(H::Matrix, A::SparseMatrixCSC, c)
    rows = rowvals(A); vals = nonzeros(A)
    @inbounds for col in axes(A, 2), n in nzrange(A, col)
        H[rows[n], col] += c * vals[n]
    end
    return H
end
add_scaled!(H::SparseMatrixCSC, A::AbstractMatrix, c) = (H .+= c .* A; H)

"""
    coo_entries(Hs) -> (Is, Js, Vs, Ridx)

Row and column indices, values and matrix index of the (structural) nonzeros of all matrices in `Hs`.
"""
function coo_entries(Hs::AbstractVector{<:AbstractMatrix})
    return coo_entries([H isa SparseMatrixCSC ? H : sparse(H) for H in Hs])
end

function coo_entries(Hs::AbstractVector{<:SparseMatrixCSC})
    N = sum(nnz, Hs; init=0)
    Is = Vector{Int64}(undef, N); Js = Vector{Int64}(undef, N); Ridx = Vector{Int64}(undef, N)
    Vs = Vector{promote_type(eltype.(Hs)...)}(undef, N)
    n = 0
    @inbounds for (R, H) in enumerate(Hs)
        rows = rowvals(H); vals = nonzeros(H)
        for col in axes(H, 2), p in nzrange(H, col)
            n += 1
            Is[n] = rows[p]; Js[n] = col; Vs[n] = vals[p]; Ridx[n] = R
        end
    end
    return Is, Js, Vs, Ridx
end

"""
    diagonalize(Hk::Vector{<:AbstractMatrix}; Neig=size(Hk[1], 1), target=0, method="shift-invert")

Diagonalizes a vector of Hamiltonian matrices `Hk` and returns the specified number of eigenvalues and eigenvectors for each matrix.

# Arguments:
- `Hk::Vector{<:AbstractMatrix}`: A vector of Hamiltonian matrices to be diagonalized. Each matrix typically corresponds to a different momentum `k` in a band structure calculation.
- `Neig::Int=size(Hk[1], 1)`: The number of eigenvalues and corresponding eigenvectors to compute for each Hamiltonian matrix. Defaults to the full diagonalization (`size(Hk[1], 1)`).
- `target::Real=0`: The target eigenvalue when computing eigenvalues of a sparse matrix.
- `method::String`: The method to be used for calculating eigenvalues of a sparse matrix.
- `vectors::Bool=true`: Whether the eigenvectors are computed (dense matrices); without them, `vs` is empty and the full
  diagonalization is several times faster.
- `window=nothing`: Energy window `(Emin, Emax)` of dense matrices: only the eigenvalues and eigenvectors in the window
  are computed. `Es` keeps `size(Hk[1], 1)` rows (padded with `NaN`), `vs` has as many columns as the k-point with the
  most states in the window (padded with zeros).

With fewer k-points than threads, the dense diagonalizations and the sparse LU factorizations of shift-invert use the
remaining threads for BLAS (ARPACK, which diagonalizes one k-point at a time, all threads).

# Returns:
- `Es::Matrix{Float64}`: A matrix where each column `Es[:, k]` contains the `Neig` eigenvalues of the `k`-th Hamiltonian matrix in `Hk`.
- `vs::Array{ComplexF64, 3}`: A 3D array where each `vs[:, :, k]` contains the `Neig` eigenvectors corresponding to the `k`-th Hamiltonian matrix in `Hk`. The dimensions of `vs` are `(Nε, Neig, Nk)`, where `Nε` is the size of each Hamiltonian matrix, `Neig` is the number of eigenvectors, and `Nk` is the number of Hamiltonian matrices.
"""
function diagonalize(Hk::Vector{<:AbstractMatrix}; Neig=size(Hk[1], 1), target=0, method="shift-invert", vectors=true,
                     window=nothing)
    Nε = size(Hk[1], 1)
    Nk = length(Hk)
    dense = !(Hk[1] isa SparseMatrixCSC)
    windowed = dense && !isnothing(window)
    results = Vector{Any}(undef, Nk)

    # Threads that the loop over k-points leaves idle go to BLAS, which the dense diagonalization and the sparse LU
    # factorization of shift-invert use (e.g., all threads for a single k-point). ARPACK diagonalizes one k-point at a
    # time (ARPACK_LOCK), so it always gets all threads.
    blas_threads = BLAS.get_num_threads()
    arpack = !dense && method == "shift-invert-arpack"
    factorizes = dense || method ∈ ("shift-invert", "shift-invert-arpack")
    if factorizes && (arpack || Nk < Threads.nthreads())
        BLAS.set_num_threads(max(blas_threads, arpack ? Threads.nthreads() : Threads.nthreads() ÷ Nk))
    end
    try
        Threads.@threads for k in eachindex(Hk)
            results[k] = windowed ? diagonalize(Hk[k], vectors=vectors, window=window) :
                                    diagonalize(Hk[k], Neig=Neig, target=target, method=method, vectors=vectors)
        end
    finally
        BLAS.set_num_threads(blas_threads)
    end

    # With a window, the number of states differs between the k-points
    Nrows = windowed ? Nε : Neig
    Ncols = windowed ? maximum(r -> length(r[1]), results, init=0) : Neig
    Es = fill(windowed ? NaN : 0.0, Nrows, Nk)
    vs = zeros(ComplexF64, vectors ? Nε : 0, Ncols, Nk)
    for (k, (E, v)) in enumerate(results)
        Es[1:length(E), k] = E
        vectors && (vs[:, 1:size(v, 2), k] = v)
    end
    return Es, vs
end

"""
    apply_window(Es, vs, window)

Keep only the eigenvalues (and eigenvectors) in the energy `window` of every k-point: `Es` padded with `NaN` (same number
of rows), `vs` with as many columns as the k-point with the most states in the window.
"""
function apply_window(Es, vs, window)
    inside = [findall(E -> window[1] ≤ E ≤ window[2], view(Es, :, k)) for k in axes(Es, 2)]
    Es_w = fill(NaN, size(Es)); vs_w = zeros(eltype(vs), size(vs, 1), maximum(length, inside, init=0), size(Es, 2))
    for (k, n) in enumerate(inside)
        Es_w[1:length(n), k] = Es[n, k]
        size(vs, 1) > 0 && (vs_w[:, 1:length(n), k] = vs[:, n, k])
    end
    return Es_w, vs_w
end

"""
    diagonalize(Hk::AbstractMatrix; Neig=size(Hk, 1), target=0, method="shift-invert")

Fully diagonalizes a Hermitian Hamiltonian matrix `Hk` and returns the eigenvalues and eigenvectors.

# Arguments:
- `Hk::AbstractMatrix`: A Hermitian matrix (Hamiltonian) to be diagonalized. The matrix should be square and typically complex-valued.
- `Neig::Int=size(Hk, 1)`: The number of eigenvalues and corresponding eigenvectors to compute. Not used.
- `target::Real=0`: The target eigenvalue around which to focus the computation. Not used.
- `method::String`: The method to be used for calculating eigenvalues of a sparse matrix.
- `window=nothing`: Energy window `(Emin, Emax)`: only the eigenvalues (and eigenvectors) in the window.

# Returns:
- `real_values::Vector{Float64}`: A vector containing the real parts of the eigenvalues of `Hk`. The eigenvalues are computed using the Hermitian matrix, so they are guaranteed to be real.
- `eigenvectors::Matrix{ComplexF64}`: A matrix where each column is an eigenvector corresponding to an eigenvalue of `Hk`. The eigenvectors are computed in the standard basis and are complex-valued.
"""
function diagonalize(Hk::AbstractMatrix; Neig=size(Hk, 1), target=0, method="shift-invert", vectors=true, window=nothing)
    if abs(hermiticity_mismatch(Hk)) > 1e-5
        maxdiff, inds = findmax(abs.(Hk .- Hermitian(Hk)))
        @warn "Hamiltonian not hermitian! Maximum missmatch: $maxdiff at $inds."
    end
    # Energy window (LAPACK zheevr with a range): the eigenvectors outside the window are not back-transformed
    if !isnothing(window)
        vectors || return real.(eigvals(Hermitian(Hk), window[1], window[2])), nothing
        eig = eigen(Hermitian(Hk), window[1], window[2])
        return real.(eig.values), eig.vectors
    end
    vectors || return real.(eigvals(Hermitian(Hk))), nothing
    eig = eigen(Hermitian(Hk))
    return real.(eig.values), eig.vectors
end

"""
    hermiticity_mismatch(H)

Return `sum(H .- Hermitian(H))` without allocating: `Hermitian(H)` differs from `H` only in the lower triangle
(where it is the conjugate of the upper triangle) and on the diagonal (where it is real).
"""
function hermiticity_mismatch(H::AbstractMatrix)
    s = zero(complex(eltype(H)))
    @inbounds for j in axes(H, 2)
        s += H[j, j] - real(H[j, j])
        for i in j+1:size(H, 1)
            s += H[i, j] - conj(H[j, i])
        end
    end
    return s
end

# Arpack keeps internal state between its reverse-communication calls and is not thread-safe, so concurrent calls
# (e.g. from the threaded loop over k-points) are serialized.
const ARPACK_LOCK = ReentrantLock()

"""
    diagonalize(Hk::SparseMatrixCSC; Neig=6, target=0, method="shift-invert", vectors=true)

Compute `Neig` eigenvalues of the sparse Hermitian matrix `Hk` close to `target` and their eigenvectors.

# Arguments:
- `Hk::SparseMatrixCSC`: A sparse Hermitian matrix.
- `Neig::Int=6`: The number of eigenvalues and eigenvectors.
- `target::Real=0`: The energy around which the eigenvalues are computed.
- `method::String`:
  - `shift-invert` (default): Hermitian Lanczos of KrylovKit on (Hk - target)⁻¹ with a sparse LU factorization, as on
    the GPU: orthonormal eigenvectors also for degenerate eigenvalues (e.g., Kramers pairs), thread-safe, so k-points
    are diagonalized in parallel.
  - `shift-invert-arpack`: ARPACK on (Hk - target)⁻¹. ARPACK has no Hermitian driver for complex matrices: its
    eigenvectors of degenerate eigenvalues are not orthogonal to each other (the eigenvalues are exact).
  - `krylov-schur`: Lanczos of KrylovKit on Hk, without factorization (slow for interior eigenvalues).

# Returns:
- `eigenvalues::Vector{Float64}`: The `Neig` eigenvalues closest to `target` (ascending order with the KrylovKit methods,
  as returned by ARPACK with `shift-invert-arpack`).
- `eigenvectors::Matrix{ComplexF64}`: The corresponding eigenvectors (columns).
"""
function diagonalize(Hk::SparseMatrixCSC; Neig=6, target=0, method="shift-invert", vectors=true)
    if method == "shift-invert-arpack"
        Es, vs, _, _, _, _ = lock(() -> eigs(Hk, nev=Neig, sigma=target), ARPACK_LOCK)
        if abs(Es[1] - target) > abs(Es[end] - target)
            return real.(Es[end-Neig+1:end]), vs[:, end-Neig+1:end]
        else
            return real.(Es[1:Neig]), vs[:, 1:Neig]
        end
    end
    krylovdim = max(30, 2Neig + 10)      # the KrylovKit default (30) is too small for more than about 25 eigenvalues
    x0 = rand(ComplexF64, size(Hk, 1))
    if method == "shift-invert"
        F = shifted_factorization(Hk, target)
        μs, xs, info = eigsolve(x -> F \ x, x0, Neig, :LM, ishermitian=true, krylovdim=krylovdim, maxiter=500)
        check_convergence(info, Neig, method)
        λs = target .+ 1 ./ real.(μs)
    elseif method == "krylov-schur"
        λs, xs, info = eigsolve(Hk, x0, Neig, EigSorter(λ -> abs(target - λ), rev=false), ishermitian=true, krylovdim=krylovdim,
                                maxiter=500)
        check_convergence(info, Neig, method)
        λs = real.(λs)
    else
        error("Unknown diag_method '$method'. Valid options are shift-invert, shift-invert-arpack and krylov-schur.")
    end
    length(λs) ≥ Neig || error("Sparse eigensolver converged only $(length(λs)) of $Neig eigenvalues.")
    closest = sortperm(abs.(λs .- target))[1:Neig]
    perm = closest[sortperm(λs[closest])]
    return λs[perm], reduce(hcat, xs[perm])
end

"""
    check_convergence(info, Neig, method)

Warn if KrylovKit converged fewer than `Neig` eigenvalues (the others are returned as unconverged Ritz values).
"""
function check_convergence(info, Neig, method)
    info.converged ≥ Neig && return nothing
    @warn "diag_method = $method: only $(info.converged) of $Neig eigenvalues converged after $(info.numiter) restarts " *
          "(largest residual $(maximum(info.normres[1:min(end, Neig)]))). Increase neig or use another diag_method."
end

"""
    shifted_factorization(Hk, target)

Sparse LU factorization of `Hk - target I`. If `target` is (numerically) an eigenvalue, it is shifted by 1e-6 eV.
"""
function shifted_factorization(Hk, target)
    try
        return lu(Hk - target * I)
    catch err
        err isa SingularException || rethrow()
        return lu(Hk - (target + 1e-6) * I)
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
    droptol!(H::Vector{AbstractSparseMatrix}, tol=1e-10)

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

Since `apply_spin_basis` places `H` in the two spin-diagonal blocks, the gradient is the sum of the spin-diagonal blocks of
`dHr`; the spin-off-diagonal blocks do not depend on `H`.

# Arguments
- `dHr::AbstractMatrix`: An abstract matrix containing the gradient data to be transformed. It should have a shape that is compatible with the spin basis operations.
- `alternating_order::Bool`: A flag that determines the order of the spin basis application. If `false`, the function applies the transformation as `kron(H, I_spin)`. If `true`, it applies the transformation as `kron(I_spin, H)`.
"""
function gradient_apply_spin_basis(dHr::AbstractMatrix; alternating_order=false)
    Nε = size(dHr, 1) ÷ 2
    if !alternating_order
        # Case 1: kron(H, I_spin)
        return Matrix(dHr[1:2:end, 1:2:end] + dHr[2:2:end, 2:2:end])
    else
        # Case 2: kron(I_spin, H)
        return Matrix(dHr[1:Nε, 1:Nε] + dHr[Nε+1:end, Nε+1:end])
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