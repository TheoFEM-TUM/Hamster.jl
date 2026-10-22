# Electronic solvers of the MD: the Mermin free energy F = Ω + μN of the electrons in the Hamiltonian at the electronic
# temperature kT, and the density matrix ρ = f(H) on the TB pairs, whose contraction with the derivatives of the matrix
# elements gives the forces (Hellmann-Feynman: dΩ = Tr[f(H) dH] at fixed μ).

"""
    ElectronResult

Result of an electronic solver: Mermin free energy `F`, chemical potential `μ`, weights of the pair gradients `W` (the
density matrix of every pair times the spin factor, block layout of the TB model), electrons per atom `w`, and
eigenvalues (if requested; a matrix with one column per k-point).
"""
struct ElectronResult
    F :: Float64
    μ :: Float64
    W :: Vector{Float64}
    w :: Vector{Float64}
    eigenvalues :: Matrix{Float64}
end

abstract type MDElectronSolver end

# ======================================================================================================================
# Exact diagonalization
# ======================================================================================================================

"""
    ExactSolver(kgrid)

Dense diagonalization at the k-points of the Γ-centered grid `kgrid` (all points of the grid, equal weights). Scales
with the cube of the number of orbitals; for the small cells of DFT reference data.
"""
struct ExactSolver <: MDElectronSolver
    ks :: Vector{SVector{3, Float64}}
    wk :: Vector{Float64}
end

function ExactSolver(kgrid::AbstractVector{<:Integer})
    ks = vec([SVector{3, Float64}((i - 1) / kgrid[1], (j - 1) / kgrid[2], (l - 1) / kgrid[3]) for i in 1:kgrid[1], j in 1:kgrid[2], l in 1:kgrid[3]])
    return ExactSolver(ks, fill(1 / length(ks), length(ks)))
end

"""
    solve_electrons!(solver, pat, tb, hblk, Δε, nz, nelec, kT; eigenvalues=0)

Free energy, chemical potential and density-matrix weights for the TB blocks `hblk` and onsite shifts `Δε` with `nelec`
electrons (`nz`: nonzeros of the Γ Hamiltonian, assembled by the caller for the kpm solver). With `eigenvalues > 0`, the
eigenvalues are returned as well: all of them at all k-points (exact), or that many closest to μ at Γ (kpm).
"""
function solve_electrons!(solver::ExactSolver, pat::GammaPattern, tb::MDTBModel, hblk, Δε, nz, nelec, kT; eigenvalues=0)
    s = pat.ns == 1 ? 2.0 : 1.0
    N = pat.N
    real_valued = pat.ns == 1 && all(iszero, solver.ks)
    T = real_valued ? Float64 : ComplexF64
    Nk = length(solver.ks)
    eigs = Vector{Any}(undef, Nk)
    tforeach(1:Nk) do ik
        H = dense_hamiltonian!(zeros(T, N, N), pat, tb, hblk, Δε, solver.ks[ik])
        eigs[ik] = eigen!(Hermitian(H))
    end
    Es = reduce(hcat, [e.values for e in eigs])
    nfun(μ) = s * sum(solver.wk[ik] * sum(ε -> fermi(ε, μ, kT), view(Es, :, ik)) for ik in 1:Nk) - nelec
    μ = solve_mu(nfun, minimum(Es) - 20kT, maximum(Es) + 20kT, s, 1e-10 * max(nelec, 1))
    Ω = s * sum(solver.wk[ik] * sum(ε -> grand_potential(ε, μ, kT), view(Es, :, ik)) for ik in 1:Nk)
    F = Ω + μ * nelec

    ns = pat.ns
    parts = tmap(1:Nk) do ik
        V = eigs[ik].vectors
        occ = fermi.(eigs[ik].values, μ, kT)
        ρ = (V .* occ') * V'
        W = zeros(tb.nblock); w = zeros(length(tb.types))
        c = s * solver.wk[ik]
        for p in eachindex(tb.pi)
            i, j = tb.pi[p], tb.pj[p]
            ni = tb.norb[i]
            b0 = tb.bstart[p] - 1
            phase = cispi(2 * dot(solver.ks[ik], tb.pn[p]))
            for β in 1:tb.norb[j], α in 1:ni, σ in 1:ns
                W[b0 + α + ni * (β - 1)] += c * real(ρ[pat.soff[j] + ns * (β - 1) + σ, pat.soff[i] + ns * (α - 1) + σ] * phase)
            end
        end
        for a in eachindex(tb.types), o in 1:ns*tb.norb[a]
            w[a] += c * real(ρ[pat.soff[a] + o, pat.soff[a] + o])
        end
        (W, w)
    end
    # (the names W and w are used in the tasks, so the sums get other names: assigning W here would make it shared)
    Wtot = sum(first, parts); wtot = sum(last, parts)
    return ElectronResult(F, μ, Wtot, wtot, eigenvalues > 0 ? Es : zeros(0, 0))
end

"""
    solve_mu(nfun, lo, hi, s, tol)

Chemical potential for the electron count `nfun(μ) = N(μ) - N_e`: in a gapped system the middle of the gap (between the
points where half an electron per spin channel is missing or added), where `nfun` is zero within `tol`, otherwise the
root of `nfun`. All points in the gap give the same free energy; the middle is a stable choice and the natural target
of shift-invert eigenvalues.
"""
function solve_mu(nfun, lo, hi, s, tol)
    μ_lo = bisect(μ -> nfun(μ) + s / 2, lo, hi, tol=1e-10)
    μ_hi = bisect(μ -> nfun(μ) - s / 2, lo, hi, tol=1e-10)
    μ_mid = (μ_lo + μ_hi) / 2
    abs(nfun(μ_mid)) ≤ tol && return μ_mid
    return bisect(nfun, lo, hi, tol=1e-12)
end

# ======================================================================================================================
# Kernel polynomial method (Fermi operator expansion with probing)
# ======================================================================================================================

"""
    ChebyshevTable(M, a, b; Nq=max(2M, 512))

Chebyshev-Gauss quadrature of the expansion of functions on [b - a, b + a] in `M` Chebyshev polynomials: the nodes
`ε` and the matrix `C` with `coefficients = C * f.(ε)` (see `chebyshev_coefficients`).
"""
struct ChebyshevTable
    M :: Int
    a :: Float64
    b :: Float64
    ε :: Vector{Float64}
    C :: Matrix{Float64}
end

function ChebyshevTable(M, a, b; Nq=max(2M, 512))
    θs = [π * (j - 0.5) / Nq for j in 1:Nq]
    C = [(n == 0 ? 1 : 2) * cos(n * θ) / Nq for n in 0:M-1, θ in θs]
    return ChebyshevTable(M, a, b, a .* cos.(θs) .+ b, C)
end

grand_potential_coefficients(tab::ChebyshevTable, μ, kT) = tab.C * grand_potential.(tab.ε, μ, kT)

"""
    probe_colors(types, norb, ns, positions, lattice, radius)

Probing vectors of the kpm solver: atoms of the same species are colored such that atoms of the same color are farther
apart than `radius` (greedy coloring, all periodic images), and every color of a species gets one vector per orbital
(including spin) of the species. Returns the vector of every orbital (`pcol`) and the number of vectors.
"""
function probe_colors(types, norb, ns, positions, lattice, radius)
    Nat = length(types)
    nl = NeighborList(positions, lattice, radius; rmin=1e-8, include=(a, b) -> types[a] == types[b] && norb[a] > 0)
    color = zeros(Int, Nat)
    ncolors = Dict{UInt8, Int}()
    for a in 1:Nat
        norb[a] == 0 && continue
        used = Set(color[nl.j[p]] for p in nl.first[a]:nl.first[a+1]-1)
        c = 1
        while c in used
            c += 1
        end
        color[a] = c
        ncolors[types[a]] = max(get(ncolors, types[a], 0), c)
    end
    base = Dict{UInt8, Int}(); nvec = 0
    for t in sort(collect(keys(ncolors)))
        base[t] = nvec
        nvec += ncolors[t] * ns * norb[findfirst(==(t), types)]
    end
    pcol = Int32[]
    for a in 1:Nat, o in 1:ns*norb[a]
        push!(pcol, base[types[a]] + (color[a] - 1) * ns * norb[a] + o)
    end
    return pcol, nvec, ncolors
end

"""
    KPMSolver

Linear-scaling electronic solver at Γ: the grand potential Ω(μ) = s Tr g(H) and the density matrix f(H) = g'(H) are
expanded in Chebyshev polynomials of the rescaled Hamiltonian, and the traces and the elements of the density matrix on
the TB pairs are obtained with probing vectors (`probe_colors`): one vector per color and orbital, which is the sum of
the unit vectors of the orbital on all atoms of the color. The density matrix decays exponentially with the distance
(at finite kT and in gapped systems), so the error of an element is set by the probing distance.

The density matrix is the exact derivative of the polynomial grand potential (`chebyshev_derivative`), so energies and
forces are consistent for any expansion order. The recursion runs on the device (`ArrayDevice`, e.g. a GPU) or on the
CPU threads.
"""
mutable struct KPMSolver{D} <: MDElectronSolver
    device :: D
    pcol :: Vector{Int32}
    nvec :: Int
    order :: Int
    tol :: Float64
    margin :: Float64
    nelec_tol :: Float64
    bounds :: Tuple{Float64, Float64}
    μ :: Float64
    table :: Union{Nothing, ChebyshevTable}
    work :: Union{Nothing, NamedTuple}
    npasses :: Int
    ncolors :: Dict{UInt8, Int}
    gap :: Tuple{Float64, Float64}
end

function KPMSolver(tb::MDTBModel, conf=get_empty_config(); device=CPUDevice(), radius=get_md_probe_radius(conf))
    pcol, nvec, ncolors = probe_colors(tb.types, tb.norb, nspin(tb), tb.reference, tb.lattice, radius)
    return KPMSolver{typeof(device)}(device, pcol, nvec, get_md_kpm_order(conf), get_md_kpm_tol(conf),
        get_md_spectrum_margin(conf), get_md_nelec_tol(conf), (NaN, NaN), NaN, nothing, nothing, 0, ncolors, (NaN, NaN))
end

kpm_eltype(T, conf) = lowercase(get_md_kpm_precision(conf)) == "single" ? (T <: Complex ? ComplexF32 : Float32) : T

"""
    lanczos_bounds(H; m=60, seed=1)

Estimate of the lowest and highest eigenvalue of the Hermitian matrix `H` (Lanczos with `m` steps, widened by the
residual norms of the extremal Ritz vectors).
"""
function lanczos_bounds(H; m=60, seed=1)
    n = size(H, 1)
    m = min(m, n)
    rng = Xoshiro(seed)
    v = normalize(randn(rng, eltype(H), n))
    vprev = zero(v)
    αs = Float64[]; βs = Float64[]
    β = 0.0
    for _ in 1:m
        w = H * v .- β .* vprev
        α = real(dot(v, w))
        w .-= α .* v
        push!(αs, α)
        β = norm(w)
        push!(βs, β)
        β < 1e-12 && break
        vprev, v = v, w ./ β
    end
    k = length(αs)
    F = eigen(SymTridiagonal(αs, βs[1:k-1]))
    res = abs.(βs[k] .* F.vectors[k, :])
    return F.values[1] - res[1], F.values[end] + res[end]
end

function kpm_order(solver::KPMSolver, a, kT)
    solver.order > 0 && return solver.order
    return ceil(Int, log(1 / solver.tol) / π * a / kT) + 16
end

function kpm_work(solver::KPMSolver, T, pat::GammaPattern)
    dev = solver.device
    N = pat.N
    w = solver.work
    if isnothing(w) || eltype(w.Tc) ≠ T || size(w.Tc) ≠ (solver.nvec, N) || length(w.rowval) ≠ length(pat.rowval)
        alloc() = to_device(dev, zeros(T, solver.nvec, N))
        w = (Tp=alloc(), Tc=alloc(), Tn=alloc(), R=alloc(), colptr=to_device(dev, Int32.(pat.colptr)),
             rowval=to_device(dev, Int32.(pat.rowval)), nz=to_device(dev, zeros(T, length(pat.rowval))),
             pcol=to_device(dev, solver.pcol), ρnz=to_device(dev, zeros(T, length(pat.rowval))),
             diag=to_device(dev, zeros(real(T), N, 32)))
        solver.work = w
    end
    return w
end

function solve_electrons!(solver::KPMSolver, pat::GammaPattern, tb::MDTBModel, hblk, Δε, nz, nelec, kT; eigenvalues=0,
                          T=eltype(pat))
    s = pat.ns == 1 ? 2.0 : 1.0
    # Spectral bounds: estimated at the first call and widened if the spectrum leaves them
    lo, hi = lanczos_bounds(gamma_matrix(pat, nz))
    if isnan(solver.bounds[1]) || lo < solver.bounds[1] || hi > solver.bounds[2]
        width = hi - lo
        solver.bounds = (lo - solver.margin * width, hi + solver.margin * width)
        solver.table = nothing
    end
    a = (solver.bounds[2] - solver.bounds[1]) / 2
    b = (solver.bounds[2] + solver.bounds[1]) / 2
    if isnothing(solver.table)
        solver.table = ChebyshevTable(kpm_order(solver, a, kT), a, b)
    end
    tab = solver.table
    w = kpm_work(solver, T, pat)
    copyto!(w.nz, T.(nz))

    dcoeffs(μ) = chebyshev_derivative(grand_potential_coefficients(tab, μ, kT), a)
    μ = isnan(solver.μ) ? b : solver.μ
    moments = kpm_pass!(w, tab, dcoeffs(μ))
    solver.npasses = 1
    nfun(μ) = s * dot(dcoeffs(μ), moments) - nelec
    # In a gapped system μ is put in the middle of the gap (the plateau of the electron count); the probed count can be
    # off by a few electrons, which would otherwise put μ at a band edge. Without a gap: the root of the count.
    gap = kpm_gap(tab, moments, nelec, kT, s)
    if !isnothing(gap)
        if !(gap[1] + 2kT ≤ μ ≤ gap[2] - 2kT)
            μ = (gap[1] + gap[2]) / 2
            moments = kpm_pass!(w, tab, dcoeffs(μ))
            solver.npasses = 2
        end
    elseif abs(nfun(μ)) > solver.nelec_tol * max(nelec, 1)
        μ = solve_mu(nfun, b - a, b + a, s, solver.nelec_tol * max(nelec, 1))
        moments = kpm_pass!(w, tab, dcoeffs(μ))
        solver.npasses = 2
    end
    solver.gap = isnothing(gap) ? (NaN, NaN) : gap
    solver.μ = μ
    F = s * dot(grand_potential_coefficients(tab, μ, kT), moments) + μ * nelec

    # Density matrix on the pattern, symmetrized from both probing estimates
    launch_kernel!(rho_nz_kernel!, w.ρnz, pat.N, w.R, w.pcol, w.colptr, w.rowval)
    ρnz = Array(w.ρnz)
    W = zeros(tb.nblock); wat = zeros(length(tb.types))
    gamma_weights!(W, wat, pat, tb, ρnz, s)
    Es = zeros(0, 0)
    if eigenvalues > 0
        Es = reshape(sparse_eigenvalues(gamma_matrix(pat, nz), μ, neig=eigenvalues), :, 1)
    end
    return ElectronResult(F, μ, W, wat, Es)
end

"""
    sparse_eigenvalues(H, target; neig=6)

Eigenvalues of the sparse Hermitian Hamiltonian `H` closest to `target` (shift-invert with a sparse LU factorization
and a Krylov-Schur eigensolver, with a Krylov space large enough for degenerate band edges).
"""
function sparse_eigenvalues(H, target; neig=6)
    n = size(H, 1)
    neig = min(neig, n - 2)
    F = lu(H - target * I)
    x0 = normalize(randn(Xoshiro(1), eltype(H), n))
    μs, _ = eigsolve(x -> F \ x, x0, neig, :LM, ishermitian=true, krylovdim=max(40, 3neig + 10), maxiter=200)
    λs = target .+ 1 ./ real.(μs[1:min(neig, length(μs))])
    return sort(λs)
end

"""
    kpm_gap(tab, moments, nelec, kT, s; ngrid=1000, rel_dos=0.02, max_offset=0.02)

Gap of the spectrum from the probed Chebyshev moments: the electron count `N(μ) = s Σ_q f(ε_q - μ) (Cᵀm)_q` on a grid of
μ, the intervals where its slope is below `rel_dos` times the mean density of states and that are wider than 4kT, and
of these the one whose count is closest to `nelec` (at most `max_offset` × nelec + 2s away). Returns `(lower, upper)`
or `nothing` (no gap, e.g. a metal).
"""
function kpm_gap(tab::ChebyshevTable, moments, nelec, kT, s; ngrid=1000, rel_dos=0.02, max_offset=0.02)
    v = transpose(tab.C) * moments
    μs = range(tab.b - tab.a, tab.b + tab.a, length=ngrid)
    N = [s * sum(fermi(tab.ε[q], μ, kT) * v[q] for q in eachindex(v)) for μ in μs]
    dμ = step(μs)
    thr = rel_dos * abs(N[end] - N[1]) / (2tab.a)       # fraction of the mean density of states
    flat = [abs(N[i+1] - N[i]) / dμ < thr for i in 1:ngrid-1]
    best = nothing; bestoff = Inf
    i = 1
    while i ≤ ngrid - 1
        if flat[i]
            j = i
            while j < ngrid - 1 && flat[j+1]
                j += 1
            end
            lo, hi = μs[i], μs[j+1]
            off = abs((N[i] + N[j+1]) / 2 - nelec)
            if hi - lo > 4kT && off < bestoff
                best = (lo, hi); bestoff = off
            end
            i = j + 1
        else
            i += 1
        end
    end
    (isnothing(best) || bestoff > max_offset * nelec + 2s) && return nothing
    return best
end

"""
    kpm_pass!(w, tab, d)

Chebyshev recursion of the probing vectors: accumulates `R = Σₙ dₙ Tₙ(H̃) V` (the density matrix applied to the probing
vectors) and returns the probed traces `Tr Tₙ(H̃)` (moments) of all orders.
"""
function kpm_pass!(w, tab::ChebyshevTable, d)
    M = tab.M
    α = 1 / tab.a; β = tab.b / tab.a
    T = eltype(w.Tc)
    moments = zeros(M)
    Tp, Tc, Tn = w.Tp, w.Tc, w.Tn
    launch_kernel!(probe_init_kernel!, Tc, size(Tc, 2), w.pcol)
    w.R .= T(d[1]) .* Tc
    moments[1] = size(Tc, 2)
    batch = DiagBatch(w.diag, Int[], moments)
    for n in 1:M-1
        c1, c2, c3 = n == 1 ? (α, -β, 0.0) : (2α, -2β, -1.0)
        cheb_step!(Tn, Tc, n == 1 ? Tc : Tp, w.R, w.colptr, w.rowval, w.nz, T(c1), T(c2), T(c3), T(d[n+1]), w.pcol, batch, n)
        Tp, Tc, Tn = Tc, Tn, Tp
    end
    flush_diag!(batch)
    return moments
end

# On the CPU the recursion runs over the columns in tasks; on devices with KernelAbstractions kernels.

"""
    DiagBatch

Probed traces of the Chebyshev vectors on a device, collected in the columns of `buf` and summed for several orders at
once (one transfer per batch instead of per order).
"""
struct DiagBatch{B}
    buf :: B
    orders :: Vector{Int}
    moments :: Vector{Float64}
end

function flush_diag!(batch::DiagBatch)
    isempty(batch.orders) && return nothing
    sums = Array(sum(view(batch.buf, :, 1:length(batch.orders)); dims=1))
    for (c, n) in enumerate(batch.orders)
        batch.moments[n+1] = sums[c]
    end
    empty!(batch.orders)
    return nothing
end

function cheb_step!(Tn::Matrix, Tc::Matrix, Tp::Matrix, R::Matrix, colptr, rowval, nz, c1, c2, c3, dn, pcol, batch, n)
    nv = size(Tc, 1)
    N = size(Tc, 2)
    partial = tmap(chunks(1:N, n=4 * Threads.nthreads())) do cols
        acc = zeros(eltype(Tn), nv)
        tr = 0.0
        @inbounds for j in cols
            fill!(acc, 0)
            for q in colptr[j]:colptr[j+1]-1
                h = conj(nz[q]); r = rowval[q]
                @simd for k in 1:nv
                    acc[k] += h * Tc[k, r]
                end
            end
            @simd for k in 1:nv
                t = c1 * acc[k] + c2 * Tc[k, j] + c3 * Tp[k, j]
                Tn[k, j] = t
                R[k, j] += dn * t
            end
            tr += real(Tn[pcol[j], j])
        end
        tr
    end
    batch.moments[n+1] = sum(partial)
    return nothing
end

function cheb_step!(Tn, Tc, Tp, R, colptr, rowval, nz, c1, c2, c3, dn, pcol, batch, n)
    launch_kernel!(cheb_step_kernel!, Tn, size(Tn), Tc, Tp, R, colptr, rowval, nz, c1, c2, c3, dn)
    push!(batch.orders, n)
    launch_kernel!(diag_kernel!, batch.buf, size(Tn, 2), Tn, pcol, length(batch.orders))
    length(batch.orders) == size(batch.buf, 2) && flush_diag!(batch)
    return nothing
end

@kernel function cheb_step_kernel!(Tn, @Const(Tc), @Const(Tp), R, @Const(colptr), @Const(rowval), @Const(nz), c1, c2, c3, dn)
    k, j = @index(Global, NTuple)
    acc = zero(eltype(Tn))
    @inbounds for q in colptr[j]:(colptr[j+1]-1)
        acc += conj(nz[q]) * Tc[k, rowval[q]]
    end
    @inbounds t = c1 * acc + c2 * Tc[k, j] + c3 * Tp[k, j]
    @inbounds Tn[k, j] = t
    @inbounds R[k, j] += dn * t
end

@kernel function diag_kernel!(buf, @Const(T), @Const(pcol), col)
    j = @index(Global)
    @inbounds buf[j, col] = real(T[pcol[j], j])
end

@kernel function probe_init_kernel!(T, @Const(pcol))
    j = @index(Global)
    @inbounds for k in axes(T, 1)
        T[k, j] = k == pcol[j] ? one(eltype(T)) : zero(eltype(T))
    end
end

"""
ρnz[q] for the nonzero q = (r, j) of the pattern: the average of the two probing estimates R[pcol[j], r] and
conj(R[pcol[r], j]) of the Hermitian density matrix.
"""
@kernel function rho_nz_kernel!(ρnz, @Const(R), @Const(pcol), @Const(colptr), @Const(rowval))
    j = @index(Global)
    @inbounds for q in colptr[j]:(colptr[j+1]-1)
        r = rowval[q]
        ρnz[q] = (R[pcol[j], r] + conj(R[pcol[r], j])) / 2
    end
end
