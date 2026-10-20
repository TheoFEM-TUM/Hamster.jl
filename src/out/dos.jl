# Density of states (DOS), projected DOS (PDOS) and crystal orbital Hamilton populations (COHP) computed from the
# real-space Hamiltonian, either by exact diagonalization (dense) or by the kernel polynomial method (KPM, sparse).
#
# Conventions:
# - All quantities include the spin degeneracy g (g = 2 without SOC, g = 1 with SOC, where every orbital is a spin
#   orbital), so that ∫ DOS dE equals the number of electrons the bands can hold.
# - The COHP of a pair of labels (A, B) is the bond-weighted DOS
#       COHP_AB(E) = g Σₖ wₖ Σₙ δ(E - Eₙₖ) Σ_{i ∈ A, j ∈ B or i ∈ B, j ∈ A} Re(c*ᵢₙₖ Hᵢⱼ(k) cⱼₙₖ),
#   summed over orbitals on different atoms (on-site terms are excluded). Negative values are bonding.
#   The sum over all pairs plus the on-site contributions gives E·DOS(E).

"""
    orbital_labels(strc, basis; soc=false) -> (orbitals, atoms)

Name (e.g. `"Pb-s"`, `"Br-px↑"`) and atom index of every orbital, in the order of the Hamiltonian
(same names as written by `write_orbital_basis`).
"""
function orbital_labels(strc, basis; soc=false)
    orbitals = String[]; atoms = Int64[]
    for iion in eachindex(basis.orbitals), orb in basis.orbitals[iion]
        element = number_to_element(strc.ions[iion].type)
        name = lm_to_orbital_map[(orb.type.l, orb.type.m)]
        for spin in (soc ? ("↑", "↓") : ("",))
            push!(orbitals, "$element-$name$spin")
            push!(atoms, iion)
        end
    end
    return orbitals, atoms
end

"""
    projection_label(orbital, projection, region="")

Label of an orbital name for the given projection: `"element"` (`"Pb-px↑"` → `"Pb"`), `"orbital"` (`"Pb-px↑"` →
`"Pb-px"`), `"defect"` or `"localized"` (region of the atom and element, e.g. `"shell1-Pb"`, see `defect_regions` and
`localized_regions`).
"""
function projection_label(orbital, projection, region="")
    if projection == "element"
        return String(split(orbital, "-")[1])
    elseif projection == "orbital"
        return replace(orbital, "↑" => "", "↓" => "")
    elseif projection ∈ ("defect", "localized")
        isempty(region) && error("The $projection projection needs the region of every orbital.")
        return "$region-$(split(orbital, "-")[1])"
    else
        error("Unknown DOS projection '$projection'. Valid options are element, orbital, defect and localized.")
    end
end

"""
    offsite(H, atoms)

Copy of `H` without the on-site blocks (matrix elements between orbitals of the same atom).
"""
function offsite(H::SparseMatrixCSC, atoms)
    is, js, vs = findnz(H)
    keep = [atoms[i] ≠ atoms[j] for (i, j) in zip(is, js)]
    return sparse(is[keep], js[keep], vs[keep], size(H)...)
end

function offsite(H::AbstractMatrix, atoms)
    Hoff = Matrix{ComplexF64}(H)
    for j in axes(H, 2), i in axes(H, 1)
        atoms[i] == atoms[j] && (Hoff[i, j] = 0)
    end
    return Hoff
end

"""
    cohp_pairs(Hr, atoms, label_index, labels, requested)

Unordered pairs of label indices `(a, b)` with `a ≤ b` for which the COHP is computed. With `requested` empty, all pairs
that are coupled by an off-site matrix element are used; otherwise only the requested pairs (e.g. `["Pb-Br", "Cs-Br"]`).
"""
function cohp_pairs(Hr, atoms, label_index, labels, requested)
    if !isempty(requested)
        return map(requested) do name
            parts = split(name, "-")
            length(parts) == 2 || error("COHP pairs must be given as A-B, e.g. Pb-Br (got '$name').")
            a = findfirst(==(parts[1]), labels); b = findfirst(==(parts[2]), labels)
            (isnothing(a) || isnothing(b)) && error("Unknown label in COHP pair '$name'. Available labels: $(join(labels, ", ")).")
            minmax(a, b)
        end
    end
    pairs = Set{Tuple{Int64, Int64}}()
    for H in Hr
        is, js, _ = findnz(sparse(H))
        for (i, j) in zip(is, js)
            atoms[i] ≠ atoms[j] && push!(pairs, minmax(label_index[i], label_index[j]))
        end
    end
    return sort(collect(pairs))
end

broadening_weight(dE, σ, kind) = kind == "gaussian" ? exp(-0.5 * (dE / σ)^2) / (σ * sqrt(2π)) :
                                 kind == "lorentzian" ? (σ / π) / (dE^2 + σ^2) :
                                 error("Unknown broadening '$kind'. Valid options are gaussian and lorentzian.")

"""
    cumulative_integral(x, y)

Cumulative trapezoidal integral of `y` over `x` (same length as `x`, starting at zero).
"""
function cumulative_integral(x, y)
    out = zeros(length(y))
    for n in 2:length(y)
        out[n] = out[n-1] + 0.5 * (y[n] + y[n-1]) * (x[n] - x[n-1])
    end
    return out
end

"""
    occupations(energies, weights, nelec) -> (occupations, fermi_energy)

Occupation (between 0 and 1) of every state for `nelec` electrons at zero temperature, where state `n` holds `weights[n]`
electrons (spin degeneracy times k-point weight). The Fermi energy lies in the middle of the gap if the last occupied level
is completely filled, and at the partially filled level otherwise.
"""
function occupations(energies, weights, nelec)
    order = sortperm(energies)
    occ = zeros(length(energies))
    remaining = nelec
    fermi = energies[order[end]]
    for (rank, n) in enumerate(order)
        remaining ≤ 1e-10 * nelec && (fermi = rank > 1 ? 0.5 * (energies[order[rank-1]] + energies[n]) : energies[n]; break)
        occ[n] = min(1.0, remaining / weights[n])
        remaining -= occ[n] * weights[n]
        fermi = energies[n]
    end
    remaining > 1e-8 && @warn "Only $(nelec - remaining) of $nelec electrons fit into the bands."
    return occ, fermi
end

# ==============================================================================================================
# Exact diagonalization
# ==============================================================================================================

"""
    exact_state_quantities(Hr, Rs, ks, atoms, members, pairs)

Diagonalize the Hamiltonian at every k-point and return, per k-point, the eigenvalues `E` (`Nε`), the projections of every
state onto the labels `P` (`nlabels × Nε`) and the COHP contributions of every pair `C` (`npairs × Nε`).
"""
function exact_state_quantities(Hr, Rs, ks, atoms, members, pairs)
    return tmap(axes(ks, 2)) do k
        H = Matrix(get_hamiltonian(Hr, Rs, ks[:, k:k])[1])
        E, V = eigen(Hermitian(H))
        absV = abs2.(V)
        P = reduce(vcat, [sum(absV[m, :], dims=1) for m in members])
        Hoff = offsite(H, atoms)
        C = zeros(length(pairs), length(E))
        for (p, (a, b)) in enumerate(pairs)
            A, B = members[a], members[b]
            X = Hoff[A, B] * V[B, :]
            # (A, B) and (B, A) contribute equally for a Hermitian H
            C[p, :] = (a == b ? 1 : 2) .* vec(real(sum(conj(V[A, :]) .* X, dims=1)))
        end
        (E, P, C)
    end
end

function dos_exact(Hr, Rs, ks, atoms, members, pairs, g, settings)
    states = exact_state_quantities(Hr, Rs, ks, atoms, members, pairs)
    Eall = reduce(vcat, first.(states))
    σ = settings.sigma
    emin = isnan(settings.emin) ? minimum(Eall) - 5σ : settings.emin
    emax = isnan(settings.emax) ? maximum(Eall) + 5σ : settings.emax
    energies = collect(range(emin, emax, length=settings.ne))
    wk = g / size(ks, 2)
    dos = zeros(length(energies)); pdos = zeros(length(energies), length(members)); cohp = zeros(length(energies), length(pairs))
    for (E, P, C) in states
        W = broadening_weight.(energies .- transpose(E), σ, settings.broadening)
        dos .+= wk .* vec(sum(W, dims=2))
        pdos .+= wk .* (W * transpose(P))
        cohp .+= wk .* (W * transpose(C))
    end
    result = (energies=energies, dos=dos, pdos=pdos, cohp=cohp, fermi_energy=NaN, icohp_fermi=Float64[])
    if !isnan(settings.nelec)
        occ, fermi = occupations(Eall, fill(wk, length(Eall)), settings.nelec)
        occ_k = Iterators.Stateful(occ)
        icohp = zeros(length(pairs))
        for (E, _, C) in states
            icohp .+= wk .* (C * collect(Iterators.take(occ_k, length(E))))
        end
        result = merge(result, (fermi_energy=fermi, icohp_fermi=icohp))
    end
    return result
end

# ==============================================================================================================
# Kernel polynomial method
# ==============================================================================================================

"""
    jackson_kernel(M)

Jackson damping factors gₘ for the moments m = 0, …, M-1 (g₀ = 1).
"""
function jackson_kernel(M)
    q = π / (M + 1)
    return [((M - m + 1) * cos(q * m) + sin(q * m) * cot(q)) / (M + 1) for m in 0:M-1]
end

"""
    hermitian_mul!(Y, H, X)

`Y = H X` for a Hermitian sparse matrix `H` and the vectors stored as the columns of `transpose(X)`, i.e., `X` is
Nvec × Nε, so that the entries of all vectors for one orbital are contiguous. Since `H = Hᴴ`, column `j` of `H` gives
the entries `j` of the products, so the columns are distributed over the threads without write conflicts, and `H` is
read once for all vectors.
"""
function hermitian_mul!(Y, H::SparseMatrixCSC, X)
    rows, vals = rowvals(H), nonzeros(H)
    tforeach(chunks(1:size(H, 2), n=4 * Threads.nthreads())) do cols
        for j in cols
            fill!(view(Y, :, j), 0)
            for p in nzrange(H, j)
                c = conj(vals[p]); r = rows[p]
                @inbounds @simd for k in axes(X, 1)
                    Y[k, j] += c * X[k, r]
                end
            end
        end
    end
    return Y
end

hermitian_mul(H::SparseMatrixCSC, v::AbstractVector) = vec(hermitian_mul!(similar(v, 1, length(v)), H, reshape(v, 1, :)))

"""
    kpm_moments(H̃, Hoff, members, pairs, M, Nvec; seed=1234, vectors=nothing)

Chebyshev moments μₘ = Tr[X Tₘ(H̃)] of the rescaled Hamiltonian `H̃` (spectrum within [-1, 1]), estimated with `Nvec`
random-phase vectors, for X = 1 (DOS), X = P_A (PDOS of label A) and X = P_A H P_B (COHP of the pair (A, B), with the
unscaled off-site Hamiltonian `Hoff`). With `vectors` (`Nvec × Nε`), these vectors are used instead and the sums
⟨v|X Tₘ(H̃)|v⟩ over them are returned (not averaged): for unit vectors on all orbitals of A, the exact PDOS moments of A
and the exact COHP moments of the pairs (A, B).

The recursion runs for all vectors at once, in one pass over the columns of `H̃` per moment that is distributed over the
threads: column `j` gives the contributions of orbital `j` to the moments and, with `H̃ = H̃ᴴ`, the entries `j` of
the next Chebyshev vectors. So `H̃` is read once per moment, and all threads are used independently of `Nvec`.
"""
function kpm_moments(H̃, Hoff, members, pairs, M, Nvec; seed=1234, vectors=nothing)
    Nε = size(H̃, 1)
    V = isnothing(vectors) ? Matrix{ComplexF64}(undef, Nvec, Nε) : Matrix{ComplexF64}(vectors)
    isnothing(vectors) && for ivec in 1:Nvec
        V[ivec, :] .= exp.(2π * im .* rand(Random.MersenneTwister(seed + ivec), Nε))
    end
    # P_B H P_A V for each pair (a, b), restricted to the orbitals of B
    X = zeros(ComplexF64, Nvec, Nε); HX = similar(X)
    Ws = map(pairs) do (a, b)
        fill!(X, 0); X[:, members[a]] .= V[:, members[a]]
        hermitian_mul!(HX, Hoff, X)[:, members[b]]
    end
    label = zeros(Int, Nε)
    for (l, A) in enumerate(members); label[A] .= l; end
    position = [zeros(Int, Nε) for _ in pairs]
    for (p, (_, b)) in enumerate(pairs); position[p][members[b]] .= eachindex(members[b]); end
    pairs_of = [[p for (p, (_, b)) in enumerate(pairs) if b == l] for l in eachindex(members)]
    weight = [a == b ? 1.0 : 2.0 for (a, b) in pairs]

    μ_dos = zeros(M); μ_pdos = zeros(M, length(members)); μ_cohp = zeros(M, length(pairs))
    # Tₘ₋₂, Tₘ₋₁ and Tₘ: T₀ = V, T₁ = H̃V, Tₘ₊₁ = 2H̃Tₘ - Tₘ₋₁
    t_prev = zeros(ComplexF64, Nvec, Nε); t_curr = copy(V); t_next = similar(V)
    columns = chunks(1:Nε, n=4 * Threads.nthreads())
    for m in 1:M
        parts = kpm_pass!(t_next, t_curr, t_prev, V, H̃, Ws, label, position, pairs_of, columns, length(members),
                          m == 1 ? 1.0 : 2.0, m == 1 ? 0.0 : 1.0, m < M)
        μ_dos[m] = sum(first, parts)
        μ_pdos[m, :] .= sum(x -> x[2], parts)
        μ_cohp[m, :] .= weight .* sum(x -> x[3], parts)
        m < M && ((t_prev, t_curr, t_next) = (t_curr, t_next, t_prev))
    end
    norm = isnothing(vectors) ? Nvec : 1
    return μ_dos ./ norm, μ_pdos ./ norm, μ_cohp ./ norm
end

"""
    kpm_local_moments(H̃, Hoff, members, pairs, local_labels, M; batch=32)

Exact PDOS moments of the labels `local_labels` (small sets of orbitals, e.g. the atoms around a defect) and exact COHP
moments of the pairs that contain one of them, from unit vectors on their orbitals (`batch` vectors at a time) instead
of the stochastic trace. Returns the PDOS moments (`M × length(local_labels)`), the indices of these pairs in `pairs`
and their COHP moments.
"""
function kpm_local_moments(H̃, Hoff, members, pairs, local_labels, M; batch=32)
    orbitals = reduce(vcat, members[local_labels], init=Int64[])
    local_pairs = findall(((a, b),) -> a ∈ local_labels || b ∈ local_labels, pairs)
    # The vectors act on the first label of a pair: put the local one first
    oriented = [pairs[p][1] ∈ local_labels ? pairs[p] : reverse(pairs[p]) for p in local_pairs]
    μ_pdos = zeros(M, length(members)); μ_cohp = zeros(M, length(local_pairs))
    for part in Iterators.partition(orbitals, batch)
        V = zeros(ComplexF64, length(part), size(H̃, 1))
        for (k, i) in enumerate(part); V[k, i] = 1; end
        _, μp, μc = kpm_moments(H̃, Hoff, members, oriented, M, length(part), vectors=V)
        μ_pdos .+= μp; μ_cohp .+= μc
    end
    return μ_pdos[:, local_labels], local_pairs, μ_cohp
end

"""
    kpm_pass!(t_next, t_curr, t_prev, V, H̃, Ws, label, position, pairs_of, columns, nmembers, a, b, recurse)

One step of `kpm_moments`, distributed over the threads by columns: the contributions of `t_curr` (Tₘ₋₁ V) to the DOS,
PDOS and COHP moments per chunk of columns and, if `recurse`, `t_next = a H̃ t_curr - b t_prev`. (A function of its own,
so that the closure does not capture the vectors that `kpm_moments` swaps.)
"""
function kpm_pass!(t_next, t_curr, t_prev, V, H̃, Ws, label, position, pairs_of, columns, nmembers, a, b, recurse)
    rows, vals = rowvals(H̃), nonzeros(H̃)
    Nvec = size(V, 1)
    return tmap(columns) do cols
        dos = 0.0; pdos = zeros(nmembers); cohp = zeros(length(Ws))
        @inbounds for j in cols
            s = 0.0
            for k in 1:Nvec
                s += real(conj(V[k, j]) * t_curr[k, j])
            end
            dos += s
            l = label[j]
            if l > 0
                pdos[l] += s
                for p in pairs_of[l]
                    n = position[p][j]; W = Ws[p]; c = 0.0
                    for k in 1:Nvec
                        c += real(conj(W[k, n]) * t_curr[k, j])
                    end
                    cohp[p] += c
                end
            end
            if recurse
                for k in 1:Nvec
                    t_next[k, j] = -b * t_prev[k, j]
                end
                for q in nzrange(H̃, j)
                    c = a * conj(vals[q]); r = rows[q]
                    @simd for k in 1:Nvec
                        t_next[k, j] += c * t_curr[k, r]
                    end
                end
            end
        end
        (dos, pdos, cohp)
    end
end

"""
    kpm_reconstruct(μ, energies, center, halfwidth)

Reconstruct the function ∑ₘ gₘ μₘ Tₘ(x) / (π √(1-x²)) (Jackson kernel) on the energy grid, for every column of `μ`.
"""
function kpm_reconstruct(μ, energies, center, halfwidth)
    μ = μ isa AbstractVector ? reshape(μ, :, 1) : μ
    M = size(μ, 1)
    g = jackson_kernel(M)
    out = zeros(length(energies), size(μ, 2))
    for (n, E) in enumerate(energies)
        x = (E - center) / halfwidth
        abs(x) < 1 || continue
        T = [cos(m * acos(x)) for m in 0:M-1]
        coeff = g .* T .* [m == 0 ? 1.0 : 2.0 for m in 0:M-1]
        out[n, :] .= vec(transpose(coeff) * μ) ./ (π * halfwidth * sqrt(1 - x^2))
    end
    return out
end

function dos_kpm(Hr, Rs, ks, atoms, members, pairs, g, settings; local_labels=Int64[])
    Hks = get_hamiltonian(sparse.(Hr), Rs, ks, Sparse())
    bounds = map(Hks) do H
        x0 = rand(Random.MersenneTwister(1), eltype(H), size(H, 1))
        Emax = real(eigsolve(x -> hermitian_mul(H, x), x0, 1, :LR, ishermitian=true)[1][1])
        Emin = real(eigsolve(x -> hermitian_mul(H, x), x0, 1, :SR, ishermitian=true)[1][1])
        (Emin, Emax)
    end
    Emin, Emax = minimum(first.(bounds)), maximum(last.(bounds))
    # Padding of three kernel widths (π × halfwidth / M) beyond the extreme eigenvalues, so that the broadened weight of
    # the lowest and highest states (e.g. deep defect states) lies on the default energy grid
    center = (Emin + Emax) / 2; halfwidth = (Emax - Emin) / 2 * (1 + 3π / settings.kpm_moments) / (1 - 0.005)
    emin = isnan(settings.emin) ? center - (1 - 0.005) * halfwidth : settings.emin
    emax = isnan(settings.emax) ? center + (1 - 0.005) * halfwidth : settings.emax
    energies = collect(range(emin, emax, length=settings.ne))
    wk = g / size(ks, 2)
    dos = zeros(length(energies)); pdos = zeros(length(energies), length(members)); cohp = zeros(length(energies), length(pairs))
    for H in Hks
        H̃ = (H - center * I) ./ halfwidth
        Hoff = offsite(H, atoms)
        μ_dos, μ_pdos, μ_cohp = kpm_moments(H̃, Hoff, members, pairs, settings.kpm_moments, settings.kpm_vectors)
        if !isempty(local_labels)
            μ_pdos[:, local_labels], local_pairs, μ_cohp[:, local_pairs] =
                kpm_local_moments(H̃, Hoff, members, pairs, local_labels, settings.kpm_moments)
            # Total DOS from the labels (exact local parts, stochastic rest): the PDOS add up to the DOS
            μ_dos = vec(sum(μ_pdos, dims=2))
        end
        dos .+= wk .* vec(kpm_reconstruct(μ_dos, energies, center, halfwidth))
        pdos .+= wk .* kpm_reconstruct(μ_pdos, energies, center, halfwidth)
        cohp .+= wk .* kpm_reconstruct(μ_cohp, energies, center, halfwidth)
    end
    result = (energies=energies, dos=dos, pdos=pdos, cohp=cohp, fermi_energy=NaN, icohp_fermi=Float64[])
    if !isnan(settings.nelec)
        # Tolerance for the electron count: three times the expected relative error of the stochastic trace
        tol = max(0.01, 3 / sqrt(size(Hks[1], 1) * settings.kpm_vectors))
        fermi = kpm_fermi_energy(energies, dos, settings.nelec, tol=tol)
        if !isnan(fermi)
            icohp = [interpolate_at(energies, cumulative_integral(energies, c), fermi) for c in eachcol(cohp)]
            result = merge(result, (fermi_energy=fermi, icohp_fermi=icohp))
        end
    end
    return result
end

"""
    interpolate_at(x, y, x0)

Linear interpolation of `y(x)` at `x0` (`x` sorted).
"""
function interpolate_at(x, y, x0)
    n = clamp(searchsortedfirst(x, x0), 2, length(x))
    return y[n-1] + (x0 - x[n-1]) / (x[n] - x[n-1]) * (y[n] - y[n-1])
end

"""
    kpm_fermi_energy(energies, dos, nelec; tol=0.01, gap_threshold=1e-2)

Fermi energy from a KPM DOS: the energy where the integrated DOS reaches `nelec`. Since the stochastic trace has a relative
error of about 1/√(orbitals × vectors), the integrated DOS in a gap can deviate slightly from `nelec`. Therefore, if
regions with vanishing DOS (below `gap_threshold` times its maximum) exist where the integrated DOS is within
`tol × nelec` of `nelec`, the Fermi energy is placed in the middle of the widest of them (the band gap of an insulator).
For small systems with discrete spectra, several such regions may exist; use the exact method there.
"""
function kpm_fermi_energy(energies, dos, nelec; tol=0.01, gap_threshold=1e-2)
    idos = cumulative_integral(energies, dos)
    if idos[end] < (1 - tol) * nelec
        @warn "The integrated DOS ($(idos[end])) does not reach nelec = $nelec; extend the energy grid."
        return NaN
    end
    # Gap: longest contiguous region with vanishing DOS near the electron count
    candidates = findall(n -> abs(idos[n] - nelec) ≤ tol * nelec && dos[n] < gap_threshold * maximum(dos), eachindex(dos))
    if !isempty(candidates)
        runs = Vector{UnitRange{Int64}}()
        for n in candidates
            (!isempty(runs) && last(runs[end]) == n - 1) ? (runs[end] = first(runs[end]):n) : push!(runs, n:n)
        end
        gap = runs[argmax(length.(runs))]
        length(gap) > 1 && return 0.5 * (energies[first(gap)] + energies[last(gap)])
    end
    n = findfirst(≥(nelec), idos)
    isnothing(n) && return energies[end]
    return n == 1 ? energies[1] : energies[n-1] + (nelec - idos[n-1]) / (idos[n] - idos[n-1]) * (energies[n] - energies[n-1])
end

# ==============================================================================================================
# Driver and output
# ==============================================================================================================

"""
    DOSSettings

Settings of the DOS post-processing, read from the `DOS` block of the config (see the DOS tags).
"""
Base.@kwdef struct DOSSettings
    quantities :: Vector{String} = ["dos", "pdos", "cohp"]
    method :: String = "auto"
    projection :: String = "element"
    cohp_pairs :: Vector{String} = String[]
    ne :: Int64 = 2000
    emin :: Float64 = NaN
    emax :: Float64 = NaN
    sigma :: Float64 = 0.05
    broadening :: String = "gaussian"
    kpm_moments :: Int64 = 1024
    kpm_vectors :: Int64 = 32
    exact_max_orbitals :: Int64 = 4000
    nelec :: Float64 = NaN
    primitive_poscar :: String = ""
    defect_shells :: Int64 = 2
    defect_tol :: Float64 = 1.0
    defect_separate :: Bool = true
    localized_threshold :: Float64 = 0.1
    localized_emin :: Float64 = NaN
    localized_emax :: Float64 = NaN
end

"""
    DOSSettings(settings::DOSSettings; kwargs...)

Copy of `settings` with the fields given as keyword arguments replaced.
"""
DOSSettings(settings::DOSSettings; kwargs...) = DOSSettings(; (f => getfield(settings, f) for f in fieldnames(DOSSettings))..., kwargs...)

"""
    dos_quantities(conf)

The quantities requested with the `dos` tag (`dos`, `pdos`, `cohp`; `all` for all three), validated.
"""
function dos_quantities(conf::Config)
    quantities = lowercase.(get_dos(conf))
    "all" ∈ quantities && return ["dos", "pdos", "cohp"]
    filter!(≠("none"), quantities)
    for q in quantities
        q ∈ ("dos", "pdos", "cohp") || error("Unknown value '$q' for tag dos. Valid values are dos, pdos, cohp and all.")
    end
    return quantities
end

DOSSettings(conf::Config) = DOSSettings(quantities=dos_quantities(conf), method=lowercase(get_dos_method(conf)),
    projection=lowercase(get_dos_projection(conf)), cohp_pairs=filter(p -> lowercase(p) ≠ "all", get_dos_cohp_pairs(conf)), ne=get_dos_ne(conf),
    emin=get_dos_emin(conf), emax=get_dos_emax(conf), sigma=get_dos_sigma(conf), broadening=lowercase(get_dos_broadening(conf)),
    kpm_moments=get_dos_kpm_moments(conf), kpm_vectors=get_dos_kpm_vectors(conf),
    exact_max_orbitals=get_dos_exact_max_orbitals(conf), nelec=get_dos_nelec(conf),
    primitive_poscar=isempty(get_dos_primitive_poscar(conf)) && haskey(conf, "Unfold") ? get_unfold_poscar(conf) : get_dos_primitive_poscar(conf),
    defect_shells=get_dos_defect_shells(conf), defect_tol=get_dos_defect_tol(conf), defect_separate=get_dos_defect_separate(conf),
    localized_threshold=get_dos_localized_threshold(conf),
    localized_emin=get_dos_localized_emin(conf), localized_emax=get_dos_localized_emax(conf))

"""
    compute_dos(Hr, Rs, orbitals, atoms, settings=DOSSettings(); ks=zeros(3, 1), soc=false, regions=fill("", length(orbitals)))

Compute the DOS, PDOS and COHP of a real-space Hamiltonian.

# Arguments
- `Hr`: Real-space Hamiltonian, one matrix per lattice vector (as returned by `get_hr`).
- `Rs`: Lattice vectors (in units of the lattice vectors), one per column.
- `orbitals`: Name of every orbital, e.g. `"Pb-s"` or `"Br-px↑"` (see `orbital_labels`).
- `atoms`: Atom index of every orbital (used to exclude on-site terms from the COHP).
- `settings`: `DOSSettings`.
- `ks`: k-points (in units of the reciprocal lattice vectors) with equal weights.
- `soc`: Whether the orbitals are spin orbitals (spin degeneracy 1 instead of 2).
- `regions`: Region of the atom of every orbital (`"defect"`, `"shell1"`, …, `"bulk"`, see `defect_regions`), for the
  defect projection. With the kpm method, the PDOS of all regions except the bulk is exact (unit vectors on their
  orbitals instead of the stochastic trace).

# Returns
A named tuple with `energies`, `dos`, `pdos` (energies × labels), `cohp` (energies × pairs), `labels`, `label_atoms`
(number of atoms per label), `pairs`, `method`,
and, if `settings.nelec` is set, `fermi_energy`, `idos`, `icohp` (integrated curves) and `icohp_fermi` (ICOHP at the Fermi
energy).
"""
function compute_dos(Hr, Rs, orbitals, atoms, settings=DOSSettings(); ks=zeros(3, 1), soc=false, regions=fill("", length(orbitals)))
    orbital_label = projection_label.(orbitals, settings.projection, regions)
    labels = unique(orbital_label)
    label_index = [findfirst(==(l), labels) for l in orbital_label]
    members = [findall(==(l), label_index) for l in eachindex(labels)]
    pairs = "cohp" ∈ settings.quantities ? cohp_pairs(Hr, atoms, label_index, labels, settings.cohp_pairs) : Tuple{Int64, Int64}[]
    g = soc ? 1 : 2
    method = settings.method == "auto" ? (length(orbitals) ≤ settings.exact_max_orbitals || settings.projection == "localized" ? "exact" : "kpm") : settings.method
    result = if method == "exact"
        dos_exact(Hr, Rs, ks, atoms, members, pairs, g, settings)
    elseif method == "kpm"
        local_labels = settings.projection == "defect" ? findall(l -> !startswith(l, "bulk"), labels) : Int64[]
        dos_kpm(Hr, Rs, ks, atoms, members, pairs, g, settings, local_labels=local_labels)
    else
        error("Unknown DOS method '$method'. Valid options are auto, exact and kpm.")
    end
    integrated = isnan(settings.nelec) ? (;) :
        (idos=cumulative_integral(result.energies, result.dos), icohp=reduce(hcat, [cumulative_integral(result.energies, c) for c in eachcol(result.cohp)], init=zeros(length(result.energies), 0)))
    label_atoms = [length(unique(atoms[m])) for m in members]
    return merge(result, integrated, (labels=labels, label_atoms=label_atoms, pairs=["$(labels[a])-$(labels[b])" for (a, b) in pairs],
                                      method=method))
end

"""
    write_dos(file, key, result, settings)

Write the result of `compute_dos` to the HDF5 file `file` under the group `key`.
"""
function write_dos(file, key, result, settings)
    h5open(file, "cw") do f
        haskey(f, key) && delete_object(f, key)
        g = create_group(f, key)
        g["energies"] = result.energies
        "dos" ∈ settings.quantities && (g["dos"] = result.dos)
        if "pdos" ∈ settings.quantities
            gp = create_group(g, "pdos")
            for (l, label) in enumerate(result.labels); gp[label] = result.pdos[:, l]; end
            ga = create_group(g, "label_atoms")
            for (l, label) in enumerate(result.labels); ga[label] = result.label_atoms[l]; end
        end
        haskey(result, :regions) && (g["regions"] = result.regions; g["region_defect"] = result.region_owner)
        if haskey(result, :participation)
            p = result.participation
            gp = create_group(g, "participation")
            gp["energies"] = p.energies; gp["pn"] = p.pn; gp["ratio"] = p.ratio
            gp["peak_energies"] = Float64[x.energy for x in p.peaks]; gp["peak_ratios"] = Float64[x.ratio for x in p.peaks]
            gp["peak_pn"] = Float64[x.pn for x in p.peaks]
            for (n, x) in enumerate(p.peaks); gp["peak_atoms/$n"] = x.atoms; end
        end
        if "cohp" ∈ settings.quantities
            gc = create_group(g, "cohp")
            for (p, pair) in enumerate(result.pairs); gc[pair] = result.cohp[:, p]; end
        end
        if !isnan(result.fermi_energy)
            g["fermi_energy"] = result.fermi_energy
            g["idos"] = result.idos
            if "cohp" ∈ settings.quantities
                gi = create_group(g, "icohp")
                for (p, pair) in enumerate(result.pairs); gi[pair] = result.icohp[:, p]; end
                gf = create_group(g, "icohp_fermi")
                for (p, pair) in enumerate(result.pairs); gf[pair] = result.icohp_fermi[p]; end
            end
        end
        attrs = attributes(g)
        attrs["method"] = result.method
        attrs["projection"] = settings.projection
        result.method == "exact" ? (attrs["sigma"] = settings.sigma; attrs["broadening"] = settings.broadening) :
                                   (attrs["kpm_moments"] = settings.kpm_moments; attrs["kpm_vectors"] = settings.kpm_vectors)
        isnan(settings.nelec) || (attrs["nelec"] = settings.nelec)
    end
end

"""
    structure_defect_regions(strc, basis, pc, settings; soc=false)

Region of every atom of `strc` (see `defect_regions`) from the mapping of its displaced positions onto the primitive
cell `pc` (so that defects that move in an MD trajectory are followed).
"""
function structure_defect_regions(strc, basis, pc, settings; soc=false)
    umap = UnfoldMap(strc, basis, pc.lattice, pc.rs_atom, pc.atom_types, soc=soc, tol=settings.defect_tol, displaced=true)
    levels, owner = defect_regions(strc, umap, pc.lattice, pc.rs_atom, shells=settings.defect_shells, displaced=true,
                                   separate=settings.defect_separate)
    all(==(-1), levels) && @warn "projection = defect: no defect found in the structure (all atoms are bulk)." maxlog=1
    return levels, owner
end

"""
    localized_regions(Hr, Rs, ks, strc, atoms, settings; soc=false)

Localized states of a structure and the atoms around them, from the eigenvectors at the k-points `ks` (no mapping onto a
primitive cell, so also polarons or self-trapped carriers in a perfect, distorted structure are found): the
energy-resolved participation of the atoms (see `participation`), the atoms that carry the states with a participation
ratio below `settings.localized_threshold` (see `localized_atoms`) and their neighbour shells (displaced positions).
Returns a named tuple with `energies`, `pn`, `ratio` (on the energy grid of the DOS), `peaks`, `levels` (0: atoms
of the localized states, n: shell n, -1: the rest) and `owner` (the localized region of every atom, see `neighbor_shells`).
"""
function localized_regions(Hr, Rs, ks, strc, atoms, settings; soc=false)
    states = atom_weight_states(Hr, Rs, ks, atoms)
    σ = settings.sigma
    Eall = reduce(vcat, first.(states))
    emin = isnan(settings.emin) ? minimum(Eall) - 5σ : settings.emin
    emax = isnan(settings.emax) ? maximum(Eall) + 5σ : settings.emax
    energies = collect(range(emin, emax, length=settings.ne))
    types = [ion.type for ion in strc.ions]
    pn, ratio, wa, density = participation(energies, states, types, σ, 1 / size(ks, 2))
    # Default window: the gap ± 1 eV (with nelec), all states otherwise
    window = (-Inf, Inf)
    if !isnan(settings.nelec)
        g = soc ? 1 : 2
        occ, _ = occupations(Eall, fill(g / size(ks, 2), length(Eall)), settings.nelec)
        vbm = maximum(Eall[occ .> 0.5], init=-Inf); cbm = minimum(Eall[occ .≤ 0.5], init=Inf)
        window = (vbm - 1, cbm + 1)
    end
    core, peaks = localized_atoms(energies, ratio, pn, wa, density, threshold=settings.localized_threshold,
                                  emin=isnan(settings.localized_emin) ? window[1] : settings.localized_emin,
                                  emax=isnan(settings.localized_emax) ? window[2] : settings.localized_emax)
    isempty(core) && @warn "projection = localized: no localized states found (participation ratio below $(settings.localized_threshold))." maxlog=1
    positions = [ion.pos + ion.dist for ion in strc.ions]
    # One region per localized state (the atoms of a state need not be neighbours); a state that shares atoms with an
    # earlier one joins its region
    state = zeros(Int64, length(positions))
    for p in peaks
        shared = filter(>(0), state[p.atoms])
        state[p.atoms] .= isempty(shared) ? maximum(state, init=0) + 1 : minimum(shared)
    end
    ids = unique(state[core])
    levels, owner = neighbor_shells(strc.lattice, positions, nearest_neighbor_distances(strc.lattice, positions), core,
                                    Tuple{Vector{Float64}, Float64}[], settings.defect_shells, separate=settings.defect_separate,
                                    core_defects=[findfirst(==(state[a]), ids) for a in core])
    return (energies=energies, pn=pn, ratio=ratio, peaks=peaks, levels=levels, owner=owner)
end

"""
    run_dos(ham, strcs, bases, local_inds, comm, conf=get_empty_config(); rank=0, nranks=1, ks=get_kpoints_from_config(conf))

Post-processing step of a standard run: compute the DOS, PDOS and COHP of every structure from its real-space Hamiltonian
and write them to the file `filename` of the `DOS` block (default `dos.h5`), one group per structure.
"""
function run_dos(ham, strcs, bases, local_inds, comm, conf=get_empty_config(); rank=0, nranks=1, ks=get_kpoints_from_config(conf),
                 verbosity=get_verbosity(conf))
    settings = DOSSettings(conf)
    filename = get_dos_filename(conf)
    soc = get_soc(conf)
    settings.projection == "localized" && settings.method == "kpm" &&
        error("projection = localized needs the eigenvectors: use method = exact (or auto).")
    settings.projection == "defect" && isempty(settings.primitive_poscar) &&
        error("projection = defect needs the primitive cell: set primitive_poscar in the DOS block (or poscar in an Unfold block).")
    pc = settings.projection == "defect" ? read_poscar(settings.primitive_poscar) : nothing
    results = map(eachindex(strcs)) do index
        system, config_index = get_system_and_config_index(index, local_inds)
        orbitals, atoms = orbital_labels(strcs[index], bases[index], soc=soc)
        localized = settings.projection == "localized" ?
            localized_regions(get_hr(ham, index), ham.Rs[index], ks, strcs[index], atoms, settings, soc=soc) : nothing
        levels, owner = !isnothing(pc) ? structure_defect_regions(strcs[index], bases[index], pc, settings, soc=soc) :
                        !isnothing(localized) ? (localized.levels, localized.owner) : (Int64[], Int64[])
        core = isnothing(localized) ? "defect" : "localized"
        ndefects = maximum(owner, init=0)
        names = [region_name(levels[a], core, owner[a], ndefects) for a in eachindex(levels)]
        regions = isempty(levels) ? fill("", length(orbitals)) : names[atoms]
        if rank == 0 && index == 1 && verbosity > 0 && !isempty(levels)
            counts = join(["$(count(==(n), names)) $n" for n in unique(names[sortperm(levels)])], ", ")
            println("DOS: $ndefects $(core == "defect" ? "defect(s)" : "localized region(s)") in structure $config_index, atoms: $counts.")
        end
        time = @elapsed result = compute_dos(get_hr(ham, index), ham.Rs[index], orbitals, atoms, settings, ks=ks, soc=soc,
                                             regions=regions)
        isempty(levels) || (result = merge(result, (regions=levels, region_owner=owner)))
        isnothing(localized) || (result = merge(result, (participation=localized,)))
        if rank == 0 && verbosity > 1; println(" DOS of structure $config_index ($(result.method)): $(round(time, digits=2)) s"); end
        (system == "" ? "$config_index" : "$(system)_$config_index", result)
    end
    # Write the structures of one rank after the other
    rank == 0 && isfile(filename) && rm(filename)
    for r in 0:nranks-1
        MPI.Barrier(comm)
        rank == r && foreach(((key, result),) -> write_dos(filename, key, result, settings), results)
    end
    MPI.Barrier(comm)
end
