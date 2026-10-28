# ======================================================================================================================
# Chebyshev-filtered subspace iteration: the eigenpairs of a sparse Hermitian matrix closest to a target energy
# ======================================================================================================================
#
# A block of trial vectors is filtered with a Chebyshev polynomial of the folded operator A = (H - σ)², whose lowest
# eigenvalues belong to the eigenvalues of H closest to the target σ. The polynomial is bounded by one on [α, β] and grows
# exponentially below α, where β ≥ max((Emin - σ)², (Emax - σ)²) follows from a short Lanczos run and α is the largest
# Ritz value of A in the current block. A Rayleigh-Ritz step with H gives the approximate eigenpairs; converged pairs are
# locked and no longer filtered. The polynomial degree of each iteration follows from the residuals and the convergence
# rate of the slowest wanted pair (as in ChASE, Winkelmann et al., ACM TOMS 45, 21 (2019)).
#
# Only products of H with blocks of vectors are needed: no factorization (memory O(nnz + m Nε)), no lock, and the same
# code runs on the CPU and on devices. Vectors are stored as the rows of m × Nε matrices, so that one pass over a row of
# H updates all vectors and the entries of all vectors for one orbital are contiguous.

"""
    HermitianCSCOperator(H)

A Hermitian `SparseMatrixCSC` for `block_matvec!`. Since `H = Hᴴ`, column `j` gives the entries `j` of the products, so the
columns are distributed over the threads without write conflicts.
"""
struct HermitianCSCOperator{M<:SparseMatrixCSC}
    H :: M
end

"""
    CSRBlockOperator(rowptr, colval, nzval)

A matrix in CSR format (device arrays) for `block_matvec!`.
"""
struct CSRBlockOperator{IV, V}
    rowptr :: IV
    colval :: IV
    nzval :: V
end

"""
    block_matvec!(Y, op, X, a, s, b1=0, Z1=X, b2=0, Z2=X)

`Y = a (H X - s X) + b1 Z1 + b2 Z2` for the vectors stored as the rows of `X`, `Y`, `Z1` and `Z2`. `Y` must not alias the
other arrays. Terms with a zero coefficient are not read.
"""
function block_matvec!(Y, op::HermitianCSCOperator, X, a, s, b1=0, Z1=X, b2=0, Z2=X)
    H = op.H
    rows, vals = rowvals(H), nonzeros(H)
    tforeach(chunks(1:size(H, 2), n=4 * Threads.nthreads())) do cols
        acc = Vector{eltype(Y)}(undef, size(X, 1))
        for j in cols
            fill!(acc, 0)
            for p in nzrange(H, j)
                c = conj(vals[p]); r = rows[p]
                @inbounds @simd for k in axes(X, 1)
                    acc[k] += c * X[k, r]
                end
            end
            @inbounds for k in axes(X, 1)
                y = a * (acc[k] - s * X[k, j])
                iszero(b1) || (y += b1 * Z1[k, j])
                iszero(b2) || (y += b2 * Z2[k, j])
                Y[k, j] = y
            end
        end
    end
    return Y
end

function block_matvec!(Y, op::CSRBlockOperator, X, a, s, b1=0, Z1=X, b2=0, Z2=X)
    launch_kernel!(csr_block_matvec_kernel!, Y, size(Y), op.rowptr, op.colval, op.nzval, X, a, s, b1, Z1, b2, Z2)
end

to_host(A) = Array(A)
to_device(proto, A) = copyto!(similar(proto, eltype(A), size(A)), A)

row_dots(X, Y) = vec(to_host(sum(conj.(X) .* Y, dims=2)))
row_norms(X) = sqrt.(vec(to_host(sum(abs2, X, dims=2))))

"""
    lanczos_spectrum(op, proto, Nε; nvec=4, steps=40, rng=Xoshiro(1))

`nvec` independent Lanczos runs of `steps` steps (without reorthogonalization) from random vectors. Returns bounds
`(Emin, Emax)` of the spectrum, widened by the residuals of the extremal Ritz pairs, and the Ritz values and weights
(squared first components of the Ritz vectors) of all runs, which sample the density of states.
"""
function lanczos_spectrum(op, proto, Nε; nvec=4, steps=40, rng=Random.Xoshiro(1))
    steps = min(steps, Nε)
    V = to_device(proto, randn(rng, eltype(proto), nvec, Nε))
    V ./= to_device(proto, row_norms(V))
    Vprev = fill!(similar(V), 0); W = similar(V)
    αs = zeros(steps, nvec); βs = zeros(steps, nvec)
    nsteps = steps
    for j in 1:steps
        block_matvec!(W, op, V, 1, 0)
        α = real.(row_dots(V, W))
        W .-= to_device(proto, α) .* V .+ to_device(proto, j > 1 ? βs[j-1, :] : zeros(nvec)) .* Vprev
        β = row_norms(W)
        αs[j, :] = α; βs[j, :] = β
        if minimum(β) < 1e-10 * maximum(abs, α)      # invariant subspace
            nsteps = j; break
        end
        W ./= to_device(proto, β)
        Vprev, V, W = V, W, Vprev
    end

    Emin, Emax = Inf, -Inf
    nodes = Float64[]; weights = Float64[]
    for v in 1:nvec
        θ, S = eigen(SymTridiagonal(αs[1:nsteps, v], βs[1:nsteps-1, v]))
        Emin = min(Emin, θ[1] - abs(βs[nsteps, v] * S[end, 1]))
        Emax = max(Emax, θ[end] + abs(βs[nsteps, v] * S[end, end]))
        append!(nodes, θ); append!(weights, abs2.(S[1, :]) ./ nvec)
    end
    return Emin, Emax, nodes, weights
end

"""
    orthonormalize_rows!(Y, L)

Orthonormalize the rows of `Y` against the (orthonormal) rows of `L` and among each other: projection and SVQB
(Stathopoulos and Wu, SIAM J. Sci. Comput. 23, 2165 (2002)), twice. Rank deficiencies, e.g. after a strong filter, are
tolerated. Returns the orthonormalized block.
"""
function orthonormalize_rows!(Y, L)
    for _ in 1:2
        size(L, 1) > 0 && (Y .-= (Y * L') * L)
        G = transpose(to_host(Y * Y'))                  # G[i, j] = ⟨yᵢ|yⱼ⟩
        d = 1 ./ sqrt.(max.(real.(diag(G)), floatmin()))
        λ, V = eigen(Hermitian(d .* G .* d'))
        λ = max.(λ, 10 * eps() * maximum(λ))
        C = d .* V ./ sqrt.(λ')                         # new vector k = Σᵢ C[i, k] yᵢ
        Y = to_device(Y, Matrix(transpose(C))) * Y
    end
    return Y
end

"""
    chebyshev_filter(op, X, deg, σ, α, β)

Apply the Chebyshev polynomial of degree `deg` of A = (H - σ)² that is bounded by one on [α, β], scaled to one at A = 0
(three-term recurrence of Zhou and Saad, J. Comput. Phys. 219, 172 (2006)), to the rows of `X`. Since the spectrum of A
lies in [0, β], the filtered vectors cannot overflow. `X` is overwritten.
"""
function chebyshev_filter(op, X, deg, σ, α, β)
    e = (β - α) / 2; c = (β + α) / 2
    s = -e / c; τ = 2 / s
    Tmp = similar(X); Y = similar(X); Ynew = similar(X)
    block_matvec!(Tmp, op, X, 1, σ)
    block_matvec!(Y, op, Tmp, s / e, σ, -s * c / e, X)
    for _ in 2:deg
        s1 = 1 / (τ - s)
        block_matvec!(Tmp, op, Y, 1, σ)
        block_matvec!(Ynew, op, Tmp, 2s1 / e, σ, -2s1 * c / e, Y, -s * s1, X)
        X, Y, Ynew = Y, Ynew, X
        s = s1
    end
    return Y
end

"""
    rayleigh_ritz_step(op, Y, L, Λ, Neig, σ, β, αest, restol, mindeg, maxdeg)

Rayleigh-Ritz step of `chebyshev_eigen` for the orthonormal block `Y` (rows, orthogonal to the locked vectors `L`): the
Ritz pairs closest to `σ` that are converged (residual below `restol`) and closer to `σ` than all unconverged ones are
locked (appended to `L` and `Λ`). Returns the remaining (active) Ritz vectors, `L`, `Λ`, and the cutoff and degree of
the next filter. `αest` is an estimate of the cutoff from the density of states, used while it is clearly above the
wanted Ritz values of A.
"""
function rayleigh_ritz_step(op, Y, L, Λ, Neig, σ, β, αest, restol, mindeg, maxdeg)
    W = similar(Y)
    block_matvec!(W, op, Y, 1, 0)

    # Ritz pairs sorted by their distance to the target
    θ, U = eigen(Hermitian(permutedims(to_host(Y * W'))))
    order = sortperm(abs.(θ .- σ))
    θ = θ[order]; Ut = to_device(Y, Matrix(transpose(U[:, order])))
    X = Ut * Y; HX = Ut * W
    res = row_norms(HX .- to_device(Y, θ) .* X)

    nconv = something(findfirst(>(restol), res), length(res) + 1) - 1
    nconv = min(nconv, Neig - length(Λ))
    if nconv > 0
        L = vcat(L, X[1:nconv, :]); Λ = vcat(Λ, θ[1:nconv])
        X = X[nconv+1:end, :]; HX = HX[nconv+1:end, :]
        θ = θ[nconv+1:end]; res = res[nconv+1:end]
    end
    length(Λ) ≥ Neig && return X, L, Λ, 0.0, 0

    # Cutoff: the largest Ritz value of A in the active block (≥ the corresponding eigenvalue of A). In the first
    # iterations, the block still contains unfiltered components and this bound is far too large; then the estimate
    # from the density of states is used as long as the wanted Ritz values lie clearly below it.
    nwant = Neig - length(Λ)
    R = HX .- σ .* X
    α = min(eigmax(Hermitian(to_host(R * R'))), 0.5 * β)
    αest > 1.2 * maximum((θ[1:nwant] .- σ) .^ 2) && (α = min(α, αest))

    # Degree: enough to converge the slowest wanted pair at its estimated rate, while the amplification within the
    # wanted pairs stays below 1e10 so that the filtered block keeps its rank
    c = (β + α) / 2; e = (β - α) / 2
    rate = [acosh(max(1.0, (c - (θ[i] - σ)^2) / e)) for i in 1:nwant]
    need = [rate[i] > 0 ? log(res[i] / restol) / rate[i] : Inf for i in 1:nwant]
    spread = maximum(rate) - minimum(rate)
    maxamp = spread > 0 ? min(maxdeg, log(1e10) / spread) : maxdeg
    deg = max(mindeg, ceil(Int, clamp(maximum(need), mindeg, maxamp)))
    return X, L, Λ, α, deg
end

"""
    chebyshev_eigen(op, proto, Nε, Neig; target=0.0, nex=max(8, Neig ÷ 2), tol=1e-10, maxiter=200, deg0=16,
                    mindeg=4, maxdeg=200, X0=nothing, seed=1, lanczos_steps=40, lanczos_vectors=4, verbosity=0)

The `Neig` eigenpairs of the Hermitian operator `op` (dimension `Nε`, see `block_matvec!`) closest to `target`, with
Chebyshev-filtered subspace iteration on a block of `Neig + nex` vectors. `proto` is an array of the element type and
device of the vectors. Pairs are converged when their residual is below `tol` times the spectral radius.

`verbosity > 1` prints the degree, block size and cutoff of every iteration.

`X0` (`k × Nε`, rows) are initial guesses for the first `k` vectors, e.g. eigenvectors of a similar matrix; the remaining
vectors are random. `deg0` is the degree of the first filter, later degrees are chosen between `mindeg` and `maxdeg`.

Returns the eigenvalues in ascending order, the eigenvectors as the rows of a `Neig × Nε` array (device of `proto`) and
a named tuple with the number of iterations and of products of `H` with a vector.
"""
function chebyshev_eigen(op, proto, Nε, Neig; target=0.0, nex=max(8, Neig ÷ 2), tol=1e-10, maxiter=200, deg0=16,
                         mindeg=4, maxdeg=200, X0=nothing, seed=1, lanczos_steps=40, lanczos_vectors=4, verbosity=0)
    T = eltype(proto)
    σ = Float64(target)
    m = min(Neig + nex, Nε)
    rng = Random.Xoshiro(seed)

    Emin, Emax, nodes, weights = lanczos_spectrum(op, proto, Nε, nvec=lanczos_vectors, steps=lanczos_steps, rng=rng)
    margin = 0.01 * (Emax - Emin)
    Emin -= margin; Emax += margin
    β = max((Emin - σ)^2, (Emax - σ)^2)
    restol = tol * max(abs(Emin), abs(Emax))
    nmatvec = lanczos_vectors * lanczos_steps

    # First cutoff: the folded energy below which the Lanczos density of states holds m states (an overestimate is safe)
    a = (nodes .- σ) .^ 2
    perm = sortperm(a)
    count = Nε .* cumsum(weights[perm])
    α = a[perm[min(something(findfirst(≥(m), count), length(perm)), length(perm))]]
    α = αest = clamp(α, 1e-6 * β, 0.5 * β)

    Xh = randn(rng, T, m, Nε)
    if !isnothing(X0)
        k0 = min(size(X0, 1), m)
        Xh[1:k0, :] = to_host(X0[1:k0, :])
    end
    L = similar(proto, T, 0, Nε)
    X = orthonormalize_rows!(to_device(proto, Xh), L)
    Λ = Float64[]
    deg = deg0
    # With an initial guess, the first cutoff and degree follow from its Ritz pairs instead of the Lanczos estimate
    if !isnothing(X0)
        X, L, Λ, α, deg = rayleigh_ritz_step(op, X, L, Λ, Neig, σ, β, αest, restol, mindeg, maxdeg)
        nmatvec += m
    end
    iter = 0
    while length(Λ) < Neig
        iter += 1
        iter > maxiter && error("Chebyshev eigensolver: $(length(Λ)) of $Neig eigenpairs converged in $maxiter iterations.")
        Y = orthonormalize_rows!(chebyshev_filter(op, X, deg, σ, α, β), L)
        nmatvec += (2deg + 1) * size(Y, 1)
        verbosity > 1 && println("  Chebyshev iteration $iter: degree $deg, active $(size(Y, 1)), cutoff $(round(α, sigdigits=3))")
        X, L, Λ, α, deg = rayleigh_ritz_step(op, Y, L, Λ, Neig, σ, β, αest, restol, mindeg, maxdeg)
    end
    verbosity > 1 && println("  Chebyshev: spectrum [$(round(Emin, digits=2)), $(round(Emax, digits=2))], $iter iterations, " *
                             "$nmatvec matrix-vector products")

    perm = sortperm(Λ)
    return Λ[perm], L[perm, :], (iterations=iter, matvecs=nmatvec)
end

"""
    chebyshev_diagonalize(H::SparseMatrixCSC, Neig, target; kwargs...)

The `Neig` eigenpairs of the Hermitian matrix `H` closest to `target` (eigenvalues in ascending order, eigenvectors as
columns). Small matrices are diagonalized densely. Keyword arguments are passed on to `chebyshev_eigen`.
"""
function chebyshev_diagonalize(H::SparseMatrixCSC, Neig, target; nex=max(8, Neig ÷ 2), kwargs...)
    Nε = size(H, 1)
    if Nε < 4 * (Neig + nex)
        E, V = eigen(Hermitian(Matrix(H)))
        closest = sort(sortperm(abs.(E .- target))[1:Neig])
        return E[closest], V[:, closest]
    end
    E, V, _ = chebyshev_eigen(HermitianCSCOperator(H), Matrix{complex(eltype(H))}(undef, 0, 0), Nε, Neig;
                              target=target, nex=nex, kwargs...)
    return E, permutedims(V)
end
