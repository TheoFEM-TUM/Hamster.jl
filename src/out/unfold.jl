# Band unfolding: the eigenstates of a supercell are projected onto the Bloch states of a primitive cell, which gives
# the effective band structure (spectral function) of distorted or disordered supercells in the primitive Brillouin zone.
#
# With the Bloch phases of Hamster (H(K) = Σ_R H(R) exp(2πi R·K), R and K in fractional coordinates), the Bloch state of
# the primitive orbital j at the primitive k-point k is, in the supercell basis at K = Mᵀk (M: supercell matrix),
#     |k, j⟩ = Nc^(-1/2) Σ_{i ∈ j} exp(2πi k·T_i) |K, i⟩,
# where the sum runs over the Nc supercell orbitals i that are copies of j, translated by the primitive lattice vector
# T_i. The spectral weight of the supercell state ψ_N at k is
#     W_N(k) = Σ_j |⟨k, j|ψ_N⟩|² = (1/Nc) Σ_j |Σ_{i ∈ j} exp(-2πi k·T_i) c_iN|²,
# which costs O(Nε) per state; the equivalent sum over all pairs i, i' ∈ j costs O(Nε Nc). The weights of all states at
# k add up to the number of primitive orbitals (full diagonalization). The weights require orthonormal eigenvectors, also
# within degenerate subspaces, so the sparse eigenvectors are computed with KrylovKit (see unfold_diag_method).

"""
    UnfoldMap

Primitive orbital `group[i]` (0: not mapped) and primitive lattice vector `T[:, i]` (fractional) of every supercell
orbital `i`; `Nj` primitive orbitals, `Nc` primitive cells in the supercell, supercell matrix `M` (`A_sc = A_pc M`).
The defects found by the mapping: `site[a]`, the primitive site of every supercell atom `a` (0: not mapped), the
`substituted` atoms (on a site of another element) and the `unmapped` atoms (interstitials, adsorbates), and the
`vacant` sites (primitive site index per vacancy, Cartesian positions `vacant_positions[:, v]` in Å).
"""
struct UnfoldMap
    group :: Vector{Int64}
    T :: Matrix{Float64}
    Nj :: Int64
    Nc :: Int64
    M :: Matrix{Int64}
    site :: Vector{Int64}
    substituted :: Vector{Int64}
    unmapped :: Vector{Int64}
    vacant :: Vector{Int64}
    vacant_positions :: Matrix{Float64}
end

"""
    cell_translations(M)

The `Nc = |det M|` primitive lattice vectors (fractional) that lie inside the supercell with supercell matrix `M`.
"""
function cell_translations(M)
    corners = [M * [i, j, k] for i in 0:1, j in 0:1, k in 0:1]
    lo = minimum(reduce(hcat, corners), dims=2); hi = maximum(reduce(hcat, corners), dims=2)
    Ts = [T for T in (collect(t) for t in Iterators.product((lo[d]:hi[d] for d in 1:3)...))
          if all(x -> -1e-8 ≤ x < 1 - 1e-8, M \ T)]
    length(Ts) == round(Int64, abs(det(M))) || error("Could not enumerate the primitive cells of the supercell.")
    return Ts
end

"""
    UnfoldMap(strc, basis, pc_lattice, pc_positions, pc_types; soc=false, tol=0.5, displaced=false)

Map the orbitals of the supercell `strc` onto the primitive cell with lattice `pc_lattice` (columns) and atoms at the
fractional `pc_positions` (columns) with types `pc_types`. The reference positions of the atoms are used, or, with
`displaced`, the positions including the distortions of an MD structure (the defects of every snapshot are found, e.g.,
a moving interstitial). Every site of the supercell holds at most one atom: the atoms are assigned, closest first, to
the nearest periodic image of a primitive site if it is closer than `tol` (Å) and not yet occupied:

- an atom of the element of the site: orbital by orbital;
- an atom of another element (substitution, antisite): its orbitals are assigned to orbitals of the same type of the
  site (in order if the orbital sets agree); orbitals without a counterpart are not mapped;
- an atom farther than `tol` from all sites (interstitial, adsorbate), or near a site that is already occupied (e.g.,
  the second atom of a split interstitial), is not mapped;
- sites without an atom (vacancies) are vacant.

Unmapped orbitals do not contribute to the spectral weights, so the weights of a defect state add up to less than one.
"""
function UnfoldMap(strc, basis, pc_lattice, pc_positions, pc_types; soc=false, tol=0.5, displaced=false)
    Mf = pc_lattice \ strc.lattice
    M = round.(Int64, Mf)
    maximum(abs, Mf - M) < 1e-3 || error("The supercell lattice is not a multiple of the primitive lattice (M = $(round.(Mf, digits=4))).")
    Nc = round(Int64, abs(det(M)))

    # Orbital types of every element (all atoms of an element have the same orbitals)
    types = Dict{UInt8, Vector{Any}}()
    for (iion, ion) in enumerate(strc.ions)
        t = [orb.type for orb in basis.orbitals[iion]]
        get!(types, ion.type, t) == t || error("Atoms of the same element have different orbitals.")
    end
    for t in unique(pc_types)
        haskey(types, t) || error("The element $(number_to_element(t)) of the primitive cell is not in the supercell (its orbitals are unknown).")
    end
    offset = cumsum([0; [length(types[t]) for t in pc_types]])
    pc_cart = pc_lattice * pc_positions

    # Nearest site image of every atom; the image is identified by the site and its translation modulo the supercell
    cell_of(T) = (f = M \ T; round.(Int64, M * (f .- floor.(f .+ 1e-8))))
    nearest = map(strc.ions) do ion
        r = displaced ? ion.pos + ion.dist : ion.pos
        best, p, Tp = Inf, 0, zeros(3)
        for q in eachindex(pc_types)
            f = pc_lattice \ (r - pc_cart[:, q])
            d = norm(pc_lattice * (f - round.(f)))
            d < best && ((best, p, Tp) = (d, q, round.(f)))
        end
        (best, p, Tp)
    end
    site = zeros(Int64, length(strc.ions)); T_atom = [zeros(3) for _ in strc.ions]
    occupied = Set{Tuple{Int64, Vector{Int64}}}()
    for a in sortperm(first.(nearest))
        d, p, Tp = nearest[a]
        key = (p, cell_of(Tp))
        (d < tol && key ∉ occupied) || continue
        push!(occupied, key); site[a] = p; T_atom[a] = Tp
    end
    any(>(0), site) || error("No supercell atom is within tol = $tol Å of a site of the primitive cell. Check the primitive cell.")
    substituted = [a for a in eachindex(site) if site[a] > 0 && strc.ions[a].type ≠ pc_types[site[a]]]
    unmapped = findall(==(0), site)

    group = Int64[]; T = Float64[]
    for (iion, ion) in enumerate(strc.ions)
        own = types[ion.type]
        p = site[iion]
        sitetypes = p > 0 ? types[pc_types[p]] : Any[]
        # Orbital of the site for every orbital of the atom (0: none)
        target = if isempty(sitetypes)
            zeros(Int64, length(own))
        elseif own == sitetypes
            collect(eachindex(own))
        else
            used = falses(length(sitetypes))
            map(eachindex(own)) do o
                h = o ≤ length(sitetypes) && !used[o] && sitetypes[o] == own[o] ? o :
                    findfirst(h -> !used[h] && sitetypes[h] == own[o], eachindex(sitetypes))
                isnothing(h) && return 0
                used[h] = true
                h
            end
        end
        for h in target, σ in (soc ? (1, 2) : (1,))
            push!(group, h == 0 ? 0 : soc ? 2 * (offset[p] + h - 1) + σ : offset[p] + h)
            append!(T, T_atom[iion])
        end
    end
    Nj = (soc ? 2 : 1) * offset[end]

    # Vacancies: site images without an atom
    vacant = Int64[]; vacant_positions = zeros(3, 0)
    for q in eachindex(pc_types), Tc in cell_translations(M)
        if (q, Tc) ∉ occupied
            push!(vacant, q); vacant_positions = hcat(vacant_positions, pc_cart[:, q] + pc_lattice * Tc)
        end
    end
    return UnfoldMap(group, reshape(T, 3, :), Nj, Nc, M, site, substituted, unmapped, vacant, vacant_positions)
end

"""
    supercell_kpoints(ks, umap; digits=8) -> (Ks, kgroups)

The supercell k-points `Ks = Mᵀ k` of the primitive k-points `ks` (fractional, columns), reduced to [0, 1) and without
duplicates; `kgroups[n]` are the primitive k-points that fold onto `Ks[:, n]`.
"""
function supercell_kpoints(ks, umap; digits=8)
    Ks_all = transpose(umap.M) * ks
    reduced(x) = (r = round(mod(x, 1), digits=digits); r == 1 ? 0.0 : r + 0.0)
    keys = [Tuple(reduced.(K)) for K in eachcol(Ks_all)]
    unique_keys = unique(keys)
    index = Dict(key => n for (n, key) in enumerate(unique_keys))
    kgroups = [Int64[] for _ in unique_keys]
    foreach(k -> push!(kgroups[index[keys[k]]], k), eachindex(keys))
    Ks = reduce(hcat, [Ks_all[:, first(group)] for group in kgroups])
    return Ks, kgroups
end

"""
    spectral_weights!(W, vs, k, umap)

Spectral weights `W[N]` of the supercell states `vs[:, N]` at the primitive k-point `k` (see the header of this file),
threaded over the states.
"""
function spectral_weights!(W, vs, k, umap)
    phase = [cis(2π * dot(k, view(umap.T, :, i))) for i in axes(umap.T, 2)]
    tforeach(chunks(axes(vs, 2), n=Threads.nthreads())) do states
        a = zeros(ComplexF64, umap.Nj)
        for N in states
            fill!(a, 0)
            @inbounds for i in axes(vs, 1)
                j = umap.group[i]
                j > 0 && (a[j] += phase[i] * vs[i, N])
            end
            W[N] = sum(abs2, a) / umap.Nc
        end
    end
    return W
end

"""
    band_weights!(Wb, vs, k, umap, X, groups)

Band-resolved spectral weights `Wb[n, N] = |⟨χₙ|ψ_N⟩|²` of the supercell states `vs[:, N]` at the primitive k-point `k`
for the band states |χₙ⟩ = Σⱼ X[j, n] |k, j⟩ of the ideal primitive cell (`X`: eigenvectors of H_pc(k)), averaged
within the degenerate `groups` of primitive bands (independent of the choice of their eigenvectors).
"""
function band_weights!(Wb, vs, k, umap, X, groups)
    phase = [cis(2π * dot(k, view(umap.T, :, i))) for i in axes(umap.T, 2)]
    Xh = Matrix(X')
    tforeach(chunks(axes(vs, 2), n=Threads.nthreads())) do states
        a = zeros(ComplexF64, umap.Nj); b = zeros(ComplexF64, umap.Nj)
        for N in states
            fill!(a, 0)
            @inbounds for i in axes(vs, 1)
                j = umap.group[i]
                j > 0 && (a[j] += phase[i] * vs[i, N])
            end
            mul!(b, Xh, a)
            for g in groups
                w = sum(abs2, view(b, g)) / (umap.Nc * length(g))
                for n in g; Wb[n, N] = w; end
            end
        end
    end
    return Wb
end

"""
    degenerate_groups(E; tol=1e-6)

Groups of indices of the sorted eigenvalues `E` that are degenerate within `tol` (eV).
"""
function degenerate_groups(E; tol=1e-6)
    groups = [[1]]
    for n in 2:length(E)
        E[n] - E[n-1] < tol ? push!(groups[end], n) : push!(groups, [n])
    end
    return groups
end

"""
    primitive_bands(conf, pc_poscar, ks; comm=MPI.COMM_SELF) -> (Es, Xs)

Bands `Es[:, k]` and eigenvectors `Xs[:, :, k]` of the primitive cell `pc_poscar` at the primitive k-points `ks`, with
the model of the run (`conf`: the same parameters and distance dependence), as reference for the band-resolved unfolding. The orbitals are in
the order of the primitive orbitals of the unfolding (atoms of the POSCAR, orbitals, spins).
"""
function primitive_bands(conf, pc_poscar, ks; comm=MPI.COMM_SELF)
    conf_pc = deepcopy(conf)
    set_value!(conf_pc, "poscar", pc_poscar)
    set_value!(conf_pc, "verbosity", 0)
    strc = Structure(conf_pc); basis = Basis(strc, conf_pc)
    ham = EffectiveHamiltonian([strc], [basis], comm, conf_pc, rank=0, nranks=1)
    Hks = get_hamiltonian(ham, 1, ks)
    Es = zeros(size(Hks[1], 1), size(ks, 2)); Xs = zeros(ComplexF64, size(Hks[1], 1), size(Hks[1], 1), size(ks, 2))
    for k in axes(ks, 2)
        E, X = eigen(Hermitian(Matrix(Hks[k])))
        Es[:, k] = E; Xs[:, :, k] = X
    end
    return Es, Xs
end

"""
    unfold_structure(ham, index, umap, ks, conf; Neig, target, method, memory, bands) -> (Es, Ws[, Wb])

Eigenvalues `Es` and spectral weights `Ws` (`Neig × Nk`) of the supercell states of structure `index` of the effective
Hamiltonian `ham` at the primitive k-points `ks`. The supercell k-points are diagonalized in batches (as many as
threads, limited to `memory` bytes of eigenvectors), with the diagonalization settings of the run (full, full in the
energy window `diag_emin`, `diag_emax`, or sparse). States outside the window have `Es = NaN` and `Ws = 0`. With the
eigenvectors `bands = Xs` of the primitive cell (see `primitive_bands`), also the band-resolved weights `Wb`
(`Nj × Neig × Nk`, see `band_weights!`).
"""
function unfold_structure(ham, index, umap, ks, conf=get_empty_config(); Neig=nothing, target=get_eig_target(conf),
                          method=unfold_diag_method(conf), window=ham.sp_diag isa Sparse ? nothing : diag_window(conf),
                          memory=0.25 * Sys.free_memory(), bands=nothing, band_energies=nothing)
    Ks, kgroups = supercell_kpoints(ks, umap)
    Nε = length(umap.group)
    Neig = something(Neig, ham.sp_diag isa Sparse ? get_neig(conf) : Nε)
    Es = zeros(Neig, size(ks, 2)); Ws = zeros(Neig, size(ks, 2))
    Wb = isnothing(bands) ? nothing : zeros(umap.Nj, Neig, size(ks, 2))
    batch = clamp(floor(Int64, memory / (16 * Nε * Neig)), 1, max(1, Threads.nthreads()))
    for Kbatch in Iterators.partition(axes(Ks, 2), batch)
        Hk = get_hamiltonian(ham, index, Ks[:, Kbatch])
        E, vs = diagonalize(Hk, Neig=Neig, target=target, method=method, vectors=true, window=window)
        for (b, n) in enumerate(Kbatch), k in kgroups[n]
            Es[:, k] = E[:, b]
            # With an energy window, only the states in the window have weights (E = NaN for the others)
            spectral_weights!(view(Ws, 1:size(vs, 2), k), view(vs, :, :, b), view(ks, :, k), umap)
            isnothing(bands) || band_weights!(view(Wb, :, 1:size(vs, 2), k), view(vs, :, :, b), view(ks, :, k), umap,
                                              view(bands, :, :, k), degenerate_groups(view(band_energies, :, k)))
        end
    end
    return isnothing(bands) ? (Es, Ws) : (Es, Ws, Wb)
end

"""
    unfold_diag_method(conf; warn=false)

Sparse eigensolver of the unfolding: `shift-invert-arpack` (ARPACK) is replaced by `shift-invert` (KrylovKit), since the
eigenvectors of ARPACK are not orthogonal within degenerate subspaces (e.g., Kramers pairs), which gives wrong spectral
weights.
"""
function unfold_diag_method(conf; warn=false)
    method = get_diag_method(conf)
    get_sp_diag(conf) isa Sparse && method == "shift-invert-arpack" || return method
    warn && @warn "Unfolding: the eigenvectors are computed with diag_method = shift-invert instead of shift-invert-arpack " *
                  "(the eigenvectors of ARPACK are not orthogonal for degenerate eigenvalues, which gives wrong spectral weights)."
    return "shift-invert"
end

"""
    spectral_function(Es, Ws, energies, σ)

Spectral function `A[e, k] = Σ_N Ws[N, k] g(energies[e] - Es[N, k])` with a normalized Gaussian `g` of width `σ`
(evaluated within ±6σ of every eigenvalue).
"""
function spectral_function(Es, Ws, energies, σ)
    A = zeros(length(energies), size(Es, 2))
    δE = step(energies)
    tforeach(axes(Es, 2)) do k
        for N in axes(Es, 1)
            Ws[N, k] > 1e-12 || continue
            e1 = max(1, floor(Int64, (Es[N, k] - 6σ - first(energies)) / δE) + 1)
            e2 = min(length(energies), ceil(Int64, (Es[N, k] + 6σ - first(energies)) / δE) + 1)
            for e in e1:e2
                A[e, k] += Ws[N, k] * exp(-0.5 * ((energies[e] - Es[N, k]) / σ)^2) / (σ * √(2π))
            end
        end
    end
    return A
end

# ==============================================================================================================
# Kernel polynomial method: the spectral function without diagonalization
# ==============================================================================================================
#
# A(k, E) = Σ_j ⟨k, j| δ(E - H(K)) |k, j⟩ is the same spectral function as Σ_N W_N(k) δ(E - E_N(k)). It is expanded in
# Chebyshev polynomials of the rescaled Hamiltonian H̃ = (H(K) - center) / halfwidth with the deterministic starting
# vectors |k, j⟩ (no stochastic trace): μₘ = Σ_j ⟨k, j|Tₘ(H̃)|k, j⟩, which needs only sparse matrix-vector products
# (linear scaling). With μ₂ₙ = 2⟨Tₙ|Tₙ⟩ - μ₀ and μ₂ₙ₊₁ = 2⟨Tₙ₊₁|Tₙ⟩ - μ₁, M moments need M/2 products.

"""
    bloch_vectors(ks, umap) -> V

Bloch states |k, j⟩ of all primitive orbitals `j` at the primitive k-points `ks` (columns) in the supercell basis, as the
rows of `V` (`(Nj × Nk) × Nε`, row `(k - 1) Nj + j`), normalized to the fraction of the `Nc` copies of `j` present in
the supercell (one without defects).
"""
function bloch_vectors(ks, umap)
    V = zeros(ComplexF64, umap.Nj * size(ks, 2), length(umap.group))
    for k in axes(ks, 2), i in axes(umap.T, 2)
        umap.group[i] > 0 || continue
        V[(k - 1) * umap.Nj + umap.group[i], i] = cis(-2π * dot(view(ks, :, k), view(umap.T, :, i))) / sqrt(umap.Nc)
    end
    return V
end

"""
    chebyshev_step!(t_next, H̃, t_curr, t_prev, columns) -> (⟨t_curr|t_curr⟩, ⟨t_next|t_curr⟩)

`t_next = 2 H̃ t_curr - t_prev` for all vectors (rows) and the scalar products of every vector, in one pass over the
columns of `H̃` (`H̃ = H̃ᴴ`: column `j` gives the entries `j` of the products) distributed over the threads.
"""
function chebyshev_step!(t_next, H̃, t_curr, t_prev, columns)
    rows, vals = rowvals(H̃), nonzeros(H̃)
    Nvec = size(t_curr, 1)
    parts = tmap(columns) do cols
        cc = zeros(Nvec); nc = zeros(Nvec)
        @inbounds for j in cols
            for v in 1:Nvec
                t_next[v, j] = -t_prev[v, j]
            end
            for q in nzrange(H̃, j)
                c = 2 * conj(vals[q]); r = rows[q]
                @simd for v in 1:Nvec
                    t_next[v, j] += c * t_curr[v, r]
                end
            end
            for v in 1:Nvec
                cc[v] += abs2(t_curr[v, j])
                nc[v] += real(conj(t_next[v, j]) * t_curr[v, j])
            end
        end
        (cc, nc)
    end
    return sum(first, parts), sum(last, parts)
end

"""
    kpm_unfold_moments(H̃, V, M) -> μ

Chebyshev moments `μ[m + 1, v] = ⟨v|Tₘ(H̃)|v⟩` (m = 0, …, M-1) of the starting vectors `V` (rows).
"""
function kpm_unfold_moments(H̃, V, M)
    Nvec, Nε = size(V)
    columns = chunks(1:Nε, n=4 * Threads.nthreads())
    μ = zeros(M, Nvec)
    t_prev = copy(V)
    t_curr = hermitian_mul!(similar(V), H̃, V)                     # T₁ V
    μ[1, :] .= vec(real(sum(abs2, V, dims=2)))
    M ≥ 2 && (μ[2, :] .= vec(real(sum(conj(V) .* t_curr, dims=2))))
    t_next = similar(V)
    n = 1
    while 2n < M
        cc, nc = chebyshev_step!(t_next, H̃, t_curr, t_prev, columns)  # t_next = Tₙ₊₁ V
        μ[2n + 1, :] .= 2 .* cc .- μ[1, :]
        2n + 1 < M && (μ[2n + 2, :] .= 2 .* nc .- μ[2, :])
        t_prev, t_curr, t_next = t_curr, t_next, t_prev
        n += 1
    end
    return μ
end

"""
    kpm_interval(Emin, Emax, M) -> (center, halfwidth)

Rescaling of the KPM with `M` moments for the spectrum [`Emin`, `Emax`]: padded by three kernel widths (π × halfwidth
/ M) and 0.5%, so that the broadened weight of the extreme states lies inside the interval and away from its edges,
where the reconstruction has the factor 1/√(1 - x²).
"""
kpm_interval(Emin, Emax, M) = ((Emin + Emax) / 2, (Emax - Emin) / 2 * (1 + 3π / M) / (1 - 0.005))

"""
    spectral_bounds(H; M=nothing, margin=0.01) -> (center, halfwidth)

Center and half width of an interval that contains the spectrum of the Hermitian sparse matrix `H` (extremal
eigenvalues from Lanczos), widened by the factor 1 + margin, or for a KPM with `M` moments, see `kpm_interval`.
"""
function spectral_bounds(H; M=nothing, margin=0.01)
    x0 = rand(Random.MersenneTwister(1), ComplexF64, size(H, 1))
    Emax = real(eigsolve(x -> hermitian_mul(H, x), x0, 1, :LR, ishermitian=true)[1][1])
    Emin = real(eigsolve(x -> hermitian_mul(H, x), x0, 1, :SR, ishermitian=true)[1][1])
    return isnothing(M) ? ((Emin + Emax) / 2, (Emax - Emin) / (2 - 2margin)) : kpm_interval(Emin, Emax, M)
end

"""
    kpm_unfold_structure(ham, index, umap, ks, M; bands=nothing, band_energies=nothing) -> (μs, centers, halfwidths[, μb])

Chebyshev moments `μs[:, k]` of the spectral function of structure `index` at the primitive k-points `ks`, with the
rescaling `centers[k]`, `halfwidths[k]` of its supercell k-point. With the eigenvectors `bands` of the primitive cell
(see `primitive_bands`), the band states |χₙ⟩ = Σⱼ X[j, n] |k, j⟩ are the starting vectors instead of the Bloch states
of the orbitals (the same space, so the same cost and total moments), and the band-resolved moments `μb[:, n, k]` are
returned as well (averaged within degenerate bands).
"""
function kpm_unfold_structure(ham, index, umap, ks, M; bands=nothing, band_energies=nothing)
    Ks, kgroups = supercell_kpoints(ks, umap)
    μs = zeros(M, size(ks, 2)); centers = zeros(size(ks, 2)); halfwidths = zeros(size(ks, 2))
    μb = isnothing(bands) ? nothing : zeros(M, umap.Nj, size(ks, 2))
    for n in axes(Ks, 2)
        Hk = get_hamiltonian(ham, index, Ks[:, n:n])
        group = kgroups[n]
        V = bloch_vectors(ks[:, group], umap)
        if !isnothing(bands)
            # Rows (g - 1) Nj + n: band state n of the primitive k-point group[g]
            for (g, k) in enumerate(group)
                rows = (g - 1) * umap.Nj + 1:g * umap.Nj
                V[rows, :] = transpose(view(bands, :, :, k)) * V[rows, :]
            end
        end
        μ, center, halfwidth = if Hk isa DeviceHk
            device_kpm_unfold(Hk, V, M)
        else
            H = sparse(first(Hk))
            c, h = spectral_bounds(H, M=M)
            kpm_unfold_moments((H - c * I) ./ h, V, M), c, h
        end
        for (g, k) in enumerate(group)
            μs[:, k] .= vec(sum(μ[:, (g - 1) * umap.Nj + 1:g * umap.Nj], dims=2))
            centers[k], halfwidths[k] = center, halfwidth
            if !isnothing(bands)
                for d in degenerate_groups(view(band_energies, :, k))
                    μb[:, d, k] .= sum(μ[:, (g - 1) * umap.Nj .+ d], dims=2) ./ length(d)
                end
            end
        end
    end
    return isnothing(bands) ? (μs, centers, halfwidths) : (μs, centers, halfwidths, μb)
end

"""
    device_kpm_unfold(Hk::DeviceHk, V, M) -> (μ, center, halfwidth)

Chebyshev moments of the starting vectors `V` (rows) for the device Hamiltonian `Hk` (one k-point) and its rescaling.
This fallback runs on the host; the CUDA extension keeps the Hamiltonian and the recursion on the GPU.
"""
function device_kpm_unfold(Hk::DeviceHk, V, M)
    H = sparse(first(host_hamiltonians(Hk)))
    center, halfwidth = spectral_bounds(H, M=M)
    return kpm_unfold_moments((H - center * I) ./ halfwidth, V, M), center, halfwidth
end

"""
    kpm_spectral_function(μs, centers, halfwidths, energies)

Spectral function `A[e, k]` from the Chebyshev moments (Jackson kernel).
"""
function kpm_spectral_function(μs, centers, halfwidths, energies)
    A = zeros(length(energies), size(μs, 2))
    tforeach(axes(μs, 2)) do k
        A[:, k] .= vec(kpm_reconstruct(μs[:, k], energies, centers[k], halfwidths[k]))
    end
    return A
end

# ==============================================================================================================
# Effective band structure
# ==============================================================================================================

"""
    effective_bands(energies, A; nbands=round(Int, maximum(sum(A, dims=1)) * step(energies)), gap_tol=0.01,
                    gap_width=0.1) -> (Eb, lower, upper)

Effective bands from the spectral function `A[e, k]` (all states, e.g. averaged over MD structures): the integrated
spectral function N_k(E) counts the primitive bands, band n holds the weight between n - 1 and n. Its energy `Eb[n, k]`
is the median of this weight (N_k = n - 1/2), `lower` and `upper` are the energies where N_k = n - 1 + 0.159 and
n - 1 + 0.841 (the central 68.2%, ±1σ for a Gaussian band).

In a distorted supercell, the weight below a gap is not an integer at every k (weight moves between k-points; only the
total over all k is conserved), which would put band edges and ribbons into the gap. Therefore N_k is snapped to the
nearest integer on every gap (energies where A < `gap_tol` states/eV over at least `gap_width` eV) and rescaled
linearly in between.
"""
function effective_bands(energies, A; nbands=round(Int, maximum(sum(A, dims=1)) * step(energies)), gap_tol=0.01,
                         gap_width=0.1)
    Eb = fill(NaN, nbands, size(A, 2)); lower = similar(Eb); upper = similar(Eb)
    crossing(N, target) = (n = findfirst(≥(target), N);
                           isnothing(n) ? NaN : n == 1 ? energies[1] :
                           energies[n-1] + (target - N[n-1]) / (N[n] - N[n-1]) * (energies[n] - energies[n-1]))
    for k in axes(A, 2)
        N = accumulate(max, cumulative_integral(energies, view(A, :, k)))    # monotone (KPM has small oscillations)
        N = snap_to_gaps(N, energies, view(A, :, k), gap_tol, gap_width)
        for b in 1:nbands
            Eb[b, k] = crossing(N, b - 0.5)
            lower[b, k] = crossing(N, b - 1 + 0.159)
            upper[b, k] = crossing(N, b - 1 + 0.841)
        end
    end
    return Eb, lower, upper
end

"""
    band_resolved_bands(energies, Ab) -> (median, lower, upper, mean, std, weight)

Effective bands from the band-resolved spectral function `Ab[e, n, k]` (see `band_weights!`): for every primitive band
n and k-point, the median and the central 68.2% (`lower`, `upper`) of its spectral function, its mean energy and
standard deviation, and its total `weight` (1 if all states are included and the band is not removed by defects).
Unlike the counting of `effective_bands`, every band has its own distribution, also where bands overlap.
"""
function band_resolved_bands(energies, Ab)
    Nj, Nk = size(Ab, 2), size(Ab, 3)
    out = [fill(NaN, Nj, Nk) for _ in 1:6]
    for k in 1:Nk, n in 1:Nj
        a = max.(view(Ab, :, n, k), 0)
        N = cumulative_integral(energies, a)
        total = N[end]
        out[6][n, k] = total
        total > 1e-6 || continue
        N ./= total
        crossing(t) = (e = findfirst(≥(t), N); e == 1 ? energies[1] :
                       energies[e-1] + (t - N[e-1]) / (N[e] - N[e-1]) * (energies[e] - energies[e-1]))
        out[1][n, k] = crossing(0.5); out[2][n, k] = crossing(0.159); out[3][n, k] = crossing(0.841)
        mean = sum(energies .* a) / sum(a)
        out[4][n, k] = mean; out[5][n, k] = sqrt(max(sum((energies .- mean) .^ 2 .* a) / sum(a), 0))
    end
    return Tuple(out)
end

"""
    gap_ranges(energies, a, tol, width) -> Vector{UnitRange}

Index ranges of the gaps of the spectral function `a`: maximal runs with `a < tol` that are at least `width` wide.
"""
function gap_ranges(energies, a, tol, width)
    gaps = UnitRange{Int}[]
    i = 1
    while i ≤ length(a)
        if a[i] < tol
            j = i
            while j < length(a) && a[j+1] < tol; j += 1; end
            energies[j] - energies[i] ≥ width && push!(gaps, i:j)
            i = j + 1
        else
            i += 1
        end
    end
    return gaps
end

"""
    snap_to_gaps(N, energies, a, tol, width)

Integrated spectral function `N` (monotone) with the value on every gap of `a` (see `gap_ranges`, not at the ends of
the energy grid) set to the nearest integer, and linearly rescaled between the gaps (and from zero at the bottom to the
rounded total at the top).
"""
function snap_to_gaps(N, energies, a, tol, width)
    knots = [(1, N[1], 0.0)]                                     # (index, value of N, snapped value)
    for gap in gap_ranges(energies, a, tol, width)
        (first(gap) == 1 || last(gap) == length(N)) && continue
        target = round((N[first(gap)] + N[last(gap)]) / 2)
        push!(knots, (first(gap), N[first(gap)], target), (last(gap), N[last(gap)], target))
    end
    push!(knots, (length(N), N[end], round(N[end])))
    out = similar(N)
    for ((i, Ni, ti), (j, Nj, tj)) in zip(knots[1:end-1], knots[2:end]), n in i:j
        out[n] = Nj > Ni ? ti + (N[n] - Ni) * (tj - ti) / (Nj - Ni) : ti
    end
    return out
end

# ==============================================================================================================
# Driver
# ==============================================================================================================

"""
    run_unfold(ham, strcs, bases, local_inds, comm, conf; rank=0, nranks=1)

Post-processing step of a standard run (switched on by an `Unfold` block): unfold the bands of every structure onto the
primitive k-points, with exact spectral weights of the supercell states (`method = exact`) or with the kernel
polynomial method (`method = kpm`), and write the spectral function averaged over all structures, its effective bands
and (with `write_weights`) the results of every structure to the file `filename` of the `Unfold` block.
"""
function run_unfold(ham, strcs, bases, local_inds, comm, conf=get_empty_config(); rank=0, nranks=1, verbosity=get_verbosity(conf))
    pc = read_poscar(get_unfold_poscar(conf))
    ks = get_kpoints_from_config(conf, kpoints_file=get_unfold_kpoints(conf))
    soc = get_soc(conf)
    filename = get_unfold_filename(conf)
    method = lowercase(get_unfold_method(conf))
    method ∈ ("exact", "kpm") || error("Unknown unfolding method '$method'. Valid options are exact and kpm.")
    M = get_unfold_kpm_moments(conf)
    method == "exact" && unfold_diag_method(conf, warn=rank == 0)
    band_resolved = get_unfold_band_resolved(conf)
    Epc, Xpc = band_resolved ? primitive_bands(conf, get_unfold_poscar(conf), ks) : (nothing, nothing)

    results = map(eachindex(strcs)) do index
        system, config_index = get_system_and_config_index(index, local_inds)
        time = @elapsed begin
            umap = UnfoldMap(strcs[index], bases[index], pc.lattice, pc.rs_atom, pc.atom_types, soc=soc, tol=get_unfold_tol(conf),
                             displaced=get_unfold_displaced_mapping(conf))
            data = method == "exact" ? unfold_structure(ham, index, umap, ks, conf, bands=Xpc, band_energies=Epc) :
                                       kpm_unfold_structure(ham, index, umap, ks, M, bands=Xpc, band_energies=Epc)
        end
        if rank == 0 && verbosity > 1; println(" Unfolding of structure $config_index ($method, $(umap.Nc) cells): $(round(time, digits=2)) s"); end
        if rank == 0 && index == 1 && verbosity > 0 && !isempty([umap.substituted; umap.unmapped; umap.vacant])
            println("Unfolding: $(length(umap.substituted)) substituted and $(length(umap.unmapped)) unmapped atoms, $(length(umap.vacant)) vacant sites " *
                    "(their states have spectral weights below one).")
        end
        (key=system == "" ? "$config_index" : "$(system)_$config_index", data=data, umap=umap)
    end

    # Energy grid common to all structures: the eigenvalues (exact) or the spectral bounds (kpm)
    σ = get_unfold_sigma(conf)
    if method == "exact"
        local_min = minimum((minimum(filter(isfinite, r.data[1]), init=Inf) for r in results), init=Inf) - 5σ
        local_max = maximum((maximum(filter(isfinite, r.data[1]), init=-Inf) for r in results), init=-Inf) + 5σ
    else
        # Inside the rescaled interval (see kpm_interval), which covers the spectrum and the kernel width
        local_min = minimum((minimum(r.data[2] .- (1 - 0.005) .* r.data[3]) for r in results), init=Inf)
        local_max = maximum((maximum(r.data[2] .+ (1 - 0.005) .* r.data[3]) for r in results), init=-Inf)
    end
    emin = isnan(get_unfold_emin(conf)) ? MPI.Allreduce(local_min, MPI.MIN, comm) : get_unfold_emin(conf)
    emax = isnan(get_unfold_emax(conf)) ? MPI.Allreduce(local_max, MPI.MAX, comm) : get_unfold_emax(conf)
    energies = range(emin, emax, length=get_unfold_ne(conf))
    spectral(r) = method == "exact" ? spectral_function(r.data[1], r.data[2], energies, σ) : kpm_spectral_function(r.data[1:3]..., energies)
    # Band-resolved spectral function (energies × bands × k-points), averaged over the structures
    function band_spectral(r)
        Ar = zeros(length(energies), size(Epc, 1), size(ks, 2))
        for n in axes(Epc, 1)
            Ar[:, n, :] = method == "exact" ? spectral_function(r.data[1], r.data[3][n, :, :], energies, σ) :
                                             kpm_spectral_function(r.data[4][:, n, :], r.data[2], r.data[3], energies)
        end
        return Ar
    end
    # kpm: weight, mean energy and width of every band from the moments (exact, independent of the energy grid):
    # ⟨H⟩ = c + h μ₁/μ₀, ⟨H²⟩ = c² + 2ch μ₁/μ₀ + h² (μ₂ + μ₀)/(2μ₀), summed over the structures with the weights μ₀
    function band_moment_sums(r)
        μb, c, h = r.data[4], reshape(r.data[2], 1, :), reshape(r.data[3], 1, :)
        m0 = μb[1, :, :]; x1 = μb[2, :, :] ./ m0; x2 = (μb[3, :, :] .+ m0) ./ (2 .* m0)
        return cat(m0, m0 .* (c .+ h .* x1), m0 .* (c .^ 2 .+ 2 .* c .* h .* x1 .+ h .^ 2 .* x2), dims=3)
    end
    moment_sums = band_resolved && method == "kpm" ?
        MPI.Reduce(reduce(+, (band_moment_sums(r) for r in results), init=zeros(size(Epc)..., 3)), +, comm, root=0) : nothing
    Ab = band_resolved ? MPI.Reduce(reduce(+, (band_spectral(r) for r in results), init=zeros(length(energies), size(Epc, 1), size(ks, 2))),
                                    +, comm, root=0) : nothing
    As = map(spectral, results)
    A = MPI.Reduce(reduce(+, As, init=zeros(length(energies), size(ks, 2))), +, comm, root=0)
    Nstrc = MPI.Reduce(length(results), +, comm, root=0)

    if rank == 0
        A ./= Nstrc
        band_resolved && (Ab ./= Nstrc)
        isfile(filename) && rm(filename)
        h5open(filename, "w") do file
            file["kpoints"] = ks
            file["energies"] = collect(energies)
            file["spectral_function"] = A
            attrs = attributes(file)
            attrs["method"] = method
            method == "exact" ? (attrs["sigma"] = σ) : (attrs["kpm_moments"] = M)
            attrs["structures"] = Nstrc
            if !isempty(results)
                u = results[1].umap
                attrs["cells"] = u.Nc; attrs["supercell_matrix"] = u.M
                attrs["substituted_atoms"] = length(u.substituted); attrs["unmapped_atoms"] = length(u.unmapped)
                attrs["vacant_sites"] = length(u.vacant)
            end
            if band_resolved
                g = create_group(file, "band_resolved")
                g["spectral_function"] = Ab; g["primitive_bands"] = Epc
                for (name, x) in zip(("median", "lower", "upper", "mean", "std", "weight"), band_resolved_bands(energies, Ab))
                    g[name] = x
                end
                if method == "kpm"
                    S0, S1, S2 = (moment_sums[:, :, i] for i in 1:3)
                    mean = S1 ./ S0
                    for (name, x) in (("weight", S0 ./ Nstrc), ("mean", mean), ("std", sqrt.(max.(S2 ./ S0 .- mean .^ 2, 0))))
                        delete_object(g, name); g[name] = x
                    end
                end
            end
            # Effective bands need all states: the integrated spectral function must reach the number of primitive bands
            if get_unfold_effective_bands(conf) && !isempty(results)
                Nj = results[1].umap.Nj
                # Weight of all states per k-point: exact for kpm (μ₀), from the spectral function for exact
                total = method == "kpm" ? vec(sum(reduce(hcat, (r.data[1][1, :] for r in results)), dims=2)) ./ length(results) :
                                          vec(sum(A, dims=1)) .* step(energies)
                if all(t -> abs(t - Nj) < 0.02 * Nj, total)
                    Eb, lower, upper = effective_bands(energies, A, nbands=Nj)
                    g = create_group(file, "effective_bands")
                    g["energies"] = Eb; g["lower"] = lower; g["upper"] = upper
                elseif verbosity > 0
                    println("Unfolding: no effective bands, the spectral function does not contain all states " *
                            "(integral $(round(minimum(total), digits=2)) of $Nj bands; use sp_diag = false or method = kpm, " *
                            "and the default emin, emax; with vacancies or unmapped orbitals, the bands cannot be counted).")
                end
            end
        end
    end
    # Results of every structure, written by one rank after the other
    if get_unfold_write_weights(conf)
        for r in 0:nranks-1
            MPI.Barrier(comm)
            rank == r && h5open(filename, "r+") do file
                for (result, Ar) in zip(results, As)
                    g = create_group(file, result.key)
                    if method == "exact"
                        g["Es"] = result.data[1]; g["Ws"] = result.data[2]
                        band_resolved && (g["band_weights"] = result.data[3])
                    else
                        g["spectral_function"] = Ar; g["moments"] = result.data[1]
                        band_resolved && (g["band_moments"] = result.data[4])
                    end
                end
            end
        end
    end
    MPI.Barrier(comm)
end
