"""
    const DEFAULT_UNCERTAINTY_NOISE = 1e-3

Default noise level λ of the Gaussian-process posterior variance (`GPVariance`), used when no `unc_noise` is given.
"""
const DEFAULT_UNCERTAINTY_NOISE = 1e-3

"""
    struct GPVariance{T}

Gaussian-process (GP) posterior variance of the kernel ridge regression model, cached as a Cholesky factorization.

With the kernel `k(x, d) = exp_sim(x, d; σ)` (so `k(x, x) = 1`) and noise level `noise` (λ), the predictive variance
at a descriptor `x` given the data points `D` is

    var(x) = k(x, x) - k(x, D) (K + λI)⁻¹ k(D, x) = 1 - ‖ L \\ v ‖²,

where `K[n, m] = k(dₙ, dₘ)`, `vₙ = k(dₙ, x)` and `L = chol(K + λI)`. It is 0 at a data point (for λ → 0) and approaches 1
far away from all data points, so it is a single-forward-pass out-of-distribution / epistemic uncertainty score for the
predicted matrix elements. It does not depend on the fitted parameters, only on the data points.

For kernel ridge regression trained with an L2 penalty λ (MSE loss on the matrix elements), this is exactly the GP
posterior variance (Rasmussen & Williams, Gaussian Processes for Machine Learning, 2006; Kanagawa et al., arXiv:1807.02582).

# Fields
- `data_points`: The kernel data points `D`.
- `σ`: The kernel width (`sim_params`).
- `noise`: The noise level λ.
- `L`: `chol(K + λI)` of the data-point kernel matrix.
"""
struct GPVariance{V}
    data_points :: V
    σ :: Float64
    noise :: Float64
    L :: Cholesky{Float64, Matrix{Float64}}
end

"""
    GPVariance(data_points, σ, noise) -> GPVariance

Build the GP posterior variance model for the kernel data points `data_points` with kernel width `σ` and noise `noise` (λ > 0).
"""
function GPVariance(data_points::V, σ::Real, noise::Real) where {T, V<:Vector{T}}
    noise > 0 || error("GPVariance: noise must be positive (got $noise); set e.g. unc_noise = 1e-3 in the ML block.")
    N = length(data_points)
    K = Matrix{Float64}(undef, N, N)
    tforeach(1:N) do n
        for m in 1:n
            K[n, m] = K[m, n] = exp_sim(data_points[n], data_points[m], σ=σ)
        end
    end
    K[diagind(K)] .+= noise
    return GPVariance(data_points, Float64(σ), Float64(noise), cholesky(Symmetric(K)))
end

"""
    gp_append(gp::GPVariance, x) -> GPVariance

Add the descriptor `x` to the data points of `gp`, updating the Cholesky factorization in O(N²).
"""
function gp_append(gp::GPVariance{V}, x::T) where {T, V<:Vector{T}}
    v = [exp_sim(d, x, σ=gp.σ) for d in gp.data_points]
    l = gp.L.L \ v
    s2 = 1 + gp.noise - dot(l, l)
    s2 > 0 || error("GPVariance: non-positive Schur complement; the noise is too small or the point duplicates a data point.")
    n = length(v)
    F = zeros(n + 1, n + 1)
    F[1:n, 1:n] .= gp.L.L
    F[n+1, 1:n] .= l
    F[n+1, n+1] = sqrt(s2)
    return GPVariance(vcat(gp.data_points, [x]), gp.σ, gp.noise, Cholesky(F, 'L', 0))
end

"""
    gp_variance(gp::GPVariance, x) -> Float64

GP posterior variance at the descriptor `x` (between 0 and 1; 1 = far from all data points).
"""
function gp_variance(gp::GPVariance, x)
    v = [exp_sim(d, x, σ=gp.σ) for d in gp.data_points]
    w = gp.L.L \ v
    return max(1 - dot(w, w), 0.0)
end

"""
    gp_variances(gp::GPVariance, X::AbstractMatrix; chunk=4096) -> Vector{Float64}

GP posterior variance of every column of `X` (a `d × n` matrix of descriptors), computed in chunks (in parallel).
"""
function gp_variances(gp::GPVariance, X::AbstractMatrix; chunk::Int=4096)
    d, n = size(X)
    length(gp.data_points) > 0 || return ones(n)
    out = Vector{Float64}(undef, n)
    tforeach(chunks(1:n, n=min(chunk, n))) do ch
        V = Matrix{Float64}(undef, length(gp.data_points), length(ch))
        for (c, t) in enumerate(ch), (r, dpoint) in enumerate(gp.data_points)
            @inbounds V[r, c] = exp_sim(dpoint, view(X, :, t), σ=gp.σ)
        end
        W = gp.L.L \ V
        for (c, t) in enumerate(ch)
            @inbounds out[t] = max(1 - dot(view(W, :, c), view(W, :, c)), 0.0)
        end
    end
    return out
end

gp_variances(gp::GPVariance, xs::AbstractVector{<:AbstractVector}) = gp_variances(gp, reduce(hcat, xs))

"""
    get_uncertainty(gp::GPVariance, h_env) -> (u, ū)

Per-source-orbital and mean prediction uncertainty of one structure with ML descriptors `h_env` (as stored in
`kernel.structure_descriptors`): `u[i]` is the sum of the GP posterior variances of all stored matrix elements with
source index `i` (onsite and outgoing hoppings, as in the per-atom score of evidential Hamiltonian models), and `ū` its
mean over the orbitals. The per-element variances are between 0 and 1; the per-orbital sums `u` are not.
"""
function get_uncertainty(gp::GPVariance, h_env::Vector{<:SparseMatrixCSC})
    Nε = size(h_env[1], 1)
    u = zeros(Nε)
    for R in eachindex(h_env)
        is, js, xs = findnz(h_env[R])
        isempty(xs) && continue
        vs = gp_variances(gp, xs)
        for (i, v) in zip(is, vs)
            u[i] += v
        end
    end
    return u, sum(u) / Nε
end

"""
    get_uncertainty(kernel::HamiltonianKernel, index; noise=DEFAULT_UNCERTAINTY_NOISE) -> (u, ū)
    get_uncertainty(kernel::HamiltonianKernelPrecalced, index; noise=DEFAULT_UNCERTAINTY_NOISE) -> (u, ū)
    get_uncertainty(ham::EffectiveHamiltonian, index; noise=DEFAULT_UNCERTAINTY_NOISE) -> (u, ū)

Single-forward-pass prediction uncertainty of the ML model for the structure `index`: the GP posterior variance of the
kernel regression (see [`GPVariance`](@ref)), aggregated per source orbital (`u`, see [`get_uncertainty`](@ref)) and its
mean `ū`. Use `atom_uncertainties(u, basis)` to aggregate per atom. `noise` is the GP noise level λ and should match the
regularization of the fit (`unc_noise` in the ML block); it only shifts the scale of the variance, not the ranking.

For `HamiltonianKernelPrecalced`, the variance is computed from the precomputed similarity features (similarities below
`sim_tol` and, with `key_dims`, similarities to data points of other key buckets are not stored), so it is slightly
overestimated there.
"""
get_uncertainty(kernel::HamiltonianKernel, index; noise=DEFAULT_UNCERTAINTY_NOISE) =
    get_uncertainty(GPVariance(kernel.data_points, kernel.sim_params, noise), kernel.structure_descriptors[index])

function get_uncertainty(kernel::HamiltonianKernelPrecalced, index; noise=DEFAULT_UNCERTAINTY_NOISE)
    gp = GPVariance(kernel.kp.datapoints, kernel.sim_params, noise)
    N_dp = length(kernel.kp.datapoints)
    desc_vec = kernel.sm.feature_vec[index]
    (NR, Ne) = kernel.sm.feature_shape[1][index]
    u = zeros(Ne)
    for R in 1:NR
        for (fv, (i, j, key)) in desc_vec[R][1]
            v = zeros(N_dp)
            v[kernel.kp.key_ranges[key]] .= fv
            w = gp.L.L \ v
            u[i] += max(1 - dot(w, w), 0.0)
        end
    end
    return u, sum(u) / Ne
end

function get_uncertainty(ham::EffectiveHamiltonian, index; kwargs...)
    for model in ham.models
        if model isa HamiltonianKernel || model isa HamiltonianKernelPrecalced
            return get_uncertainty(model, index; kwargs...)
        end
    end
    error("get_uncertainty: the effective Hamiltonian has no ML kernel model (add an ML block).")
end

"""
    atom_uncertainties(u, basis::Basis) -> Vector{Float64}

Aggregate the per-source-orbital uncertainties `u` (from [`get_uncertainty`](@ref)) per atom: the sum over the orbitals of
each ion (the per-atom score `uᵢ` used to rank structures in evidential Hamiltonian models).
"""
function atom_uncertainties(u::Vector{Float64}, basis::Basis)
    ion_of_orbital = [iion for iion in eachindex(basis.orbitals) for _ in basis.orbitals[iion]]
    length(u) == length(ion_of_orbital) || error("u has $(length(u)) entries, but the basis has $(length(ion_of_orbital)) orbitals.")
    ua = zeros(length(basis.orbitals))
    for (i, ui) in enumerate(u)
        ua[ion_of_orbital[i]] += ui
    end
    return ua
end
