# Device kernels written with KernelAbstractions. Every kernel assigns one work item per output
# element and loops over its contributions, so no atomics are needed and results are deterministic.
# Launches are stream-ordered on GPUs and blocking on the CPU, and copies to the host synchronize.

function launch_kernel!(kernel, out, ndrange, args...)
    prod(ndrange) == 0 && return out
    backend = KernelAbstractions.get_backend(out)
    kernel(backend)(out, args...; ndrange=ndrange)
    return out
end

"""
y[r] += Σ_q nzval[q] * x[colval[q]] for a matrix in CSR format.
"""
@kernel function csr_matvec_add_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x))
    r = @index(Global)
    acc = zero(eltype(y))
    @inbounds for q in rowptr[r]:(rowptr[r+1]-1)
        acc += nzval[q] * x[colval[q]]
    end
    @inbounds y[r] += acc
end

"""
y[r] = real(Σ_q nzval[q] * x[colval[q]]) for a matrix in CSR format.
"""
@kernel function csr_real_matvec_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x))
    r = @index(Global)
    acc = zero(promote_type(eltype(nzval), eltype(x)))
    @inbounds for q in rowptr[r]:(rowptr[r+1]-1)
        acc += nzval[q] * x[colval[q]]
    end
    @inbounds y[r] = real(acc)
end

@inline function squared_distance(X, t, D, n)
    d2 = zero(eltype(X))
    @inbounds for a in axes(X, 1)
        δ = X[a, t] - D[a, n]
        d2 += δ * δ
    end
    return d2
end

"""
hr[rows[t]] += Σ_n p[n] * exp(-|X[:, t] - D[:, n]|² * γ), i.e., the Gaussian kernel model with γ = 1/(2σ²).
"""
@kernel function gaussian_kernel_hr_kernel!(hr, @Const(rows), @Const(X), @Const(D), @Const(p), γ)
    t = @index(Global)
    acc = zero(eltype(p))
    @inbounds for n in axes(D, 2)
        acc += p[n] * exp(-squared_distance(X, t, D, n) * γ)
    end
    @inbounds hr[rows[t]] += acc
end

"""
g[n] = Σ_t real(G[rows[t]]) * exp(-|X[:, t] - D[:, n]|² * γ), the gradient of the Gaussian kernel model.
"""
@kernel function gaussian_kernel_grad_kernel!(g, @Const(rows), @Const(X), @Const(D), @Const(G), γ)
    n = @index(Global)
    acc = zero(eltype(g))
    @inbounds for t in axes(X, 2)
        acc += real(G[rows[t]]) * exp(-squared_distance(X, t, D, n) * γ)
    end
    @inbounds g[n] = acc
end

"""
Hk[q, k] = Σ_{e ∈ group(q)} hr[e] * phase[Ridx[e], k], where group(q) contains all entries of the real-space
Hamiltonian that contribute to the nonzero `q` of the k-space Hamiltonian.
"""
@kernel function assemble_hk_kernel!(hk, @Const(grp_ptr), @Const(grp_ent), @Const(hr), @Const(Ridx), @Const(phase))
    q, k = @index(Global, NTuple)
    acc = zero(eltype(hk))
    @inbounds for t in grp_ptr[q]:(grp_ptr[q+1]-1)
        e = grp_ent[t]
        acc += hr[e] * phase[Ridx[e], k]
    end
    @inbounds hk[q, k] = acc
end

"""
H[rows[q], cols[q], k] = nzval[q, k], used to convert a batch of sparse k-space Hamiltonians to dense matrices.
"""
@kernel function densify_kernel!(H, @Const(rows), @Const(cols), @Const(nzval))
    q, k = @index(Global, NTuple)
    @inbounds H[rows[q], cols[q], k] = nzval[q, k]
end

"""
G[e] = Σ_k phase[R, k] Σ_m w[m, k] conj(Ψ[i, m, k]) Ψ[j, m, k] with `(i, j, R)` the indices of entry `e`.
This is the Hellmann-Feynman gradient dL/dHr[R][i, j] evaluated only at the entries of the model.
"""
@kernel function hellman_feynman_kernel!(G, @Const(I), @Const(J), @Const(Ridx), @Const(phase), @Const(Ψ), @Const(w))
    e = @index(Global)
    @inbounds i = I[e]; j = J[e]; R = Ridx[e]
    acc = zero(eltype(G))
    @inbounds for k in axes(Ψ, 3)
        s = zero(eltype(G))
        for m in axes(Ψ, 2)
            s += w[m, k] * conj(Ψ[i, m, k]) * Ψ[j, m, k]
        end
        acc += phase[R, k] * s
    end
    @inbounds G[e] = acc
end
