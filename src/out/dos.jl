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
    projection_label(orbital, projection)

Label of an orbital name for the given projection: `"element"` (`"Pb-px↑"` → `"Pb"`) or `"orbital"` (`"Pb-px↑"` → `"Pb-px"`).
"""
function projection_label(orbital, projection)
    if projection == "element"
        return String(split(orbital, "-")[1])
    elseif projection == "orbital"
        return replace(orbital, "↑" => "", "↓" => "")
    else
        error("Unknown DOS projection '$projection'. Valid options are element and orbital.")
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
    kpm_moments(H̃, Hoff, members, pairs, M, Nvec; seed=1234)

Chebyshev moments μₘ = Tr[X Tₘ(H̃)] of the rescaled Hamiltonian `H̃` (spectrum within [-1, 1]), estimated with `Nvec`
random-phase vectors, for X = 1 (DOS), X = P_A (PDOS of label A) and X = P_A H P_B (COHP of the pair (A, B), with the
unscaled off-site Hamiltonian `Hoff`).
"""
function kpm_moments(H̃, Hoff, members, pairs, M, Nvec; seed=1234)
    Nε = size(H̃, 1)
    partial = tmap(chunks(1:Nvec, n=min(Nvec, Threads.nthreads()))) do vecs
        # `local`: the enclosing function assigns variables of the same names, which would otherwise be shared by all tasks
        local μ_dos = zeros(M)
        local μ_pdos = zeros(M, length(members))
        local μ_cohp = zeros(M, length(pairs))
        t_prev = zeros(ComplexF64, Nε); t_curr = similar(t_prev); t_next = similar(t_prev); x = similar(t_prev)
        for ivec in vecs
            v = exp.(2π * im .* rand(Random.MersenneTwister(seed + ivec), Nε))
            # w = P_B H P_A v, restricted to the orbitals of B
            ws = map(pairs) do (a, b)
                fill!(x, 0); x[members[a]] .= v[members[a]]
                (Hoff * x)[members[b]]
            end
            # Chebyshev recursion: T₀v = v, T₁v = H̃v, Tₘ₊₁v = 2H̃Tₘv - Tₘ₋₁v
            t_prev .= v
            mul!(t_curr, H̃, v)
            for m in 1:M
                if m ≥ 3
                    mul!(t_next, H̃, t_curr)
                    @. t_next = 2 * t_next - t_prev
                    t_prev, t_curr, t_next = t_curr, t_next, t_prev
                end
                t = m == 1 ? t_prev : t_curr    # Tₘ₋₁(H̃) v
                μ_dos[m] += real(dot(v, t))
                for (l, A) in enumerate(members)
                    μ_pdos[m, l] += real(dot(view(v, A), view(t, A)))
                end
                for (p, (a, b)) in enumerate(pairs)
                    μ_cohp[m, p] += (a == b ? 1 : 2) * real(dot(ws[p], view(t, members[b])))
                end
            end
        end
        (μ_dos, μ_pdos, μ_cohp)
    end
    μ_dos, μ_pdos, μ_cohp = (sum(getindex.(partial, i)) ./ Nvec for i in 1:3)
    return μ_dos, μ_pdos, μ_cohp
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

function dos_kpm(Hr, Rs, ks, atoms, members, pairs, g, settings)
    Hks = get_hamiltonian(sparse.(Hr), Rs, ks, Sparse())
    bounds = map(Hks) do H
        Emax = real(eigsolve(H, 1, :LR, ishermitian=true)[1][1])
        Emin = real(eigsolve(H, 1, :SR, ishermitian=true)[1][1])
        (Emin, Emax)
    end
    Emin, Emax = minimum(first.(bounds)), maximum(last.(bounds))
    center = (Emin + Emax) / 2; halfwidth = (Emax - Emin) / (2 - 0.01)
    emin = isnan(settings.emin) ? Emin : settings.emin
    emax = isnan(settings.emax) ? Emax : settings.emax
    energies = collect(range(emin, emax, length=settings.ne))
    wk = g / size(ks, 2)
    dos = zeros(length(energies)); pdos = zeros(length(energies), length(members)); cohp = zeros(length(energies), length(pairs))
    for H in Hks
        H̃ = (H - center * I) ./ halfwidth
        μ_dos, μ_pdos, μ_cohp = kpm_moments(H̃, offsite(H, atoms), members, pairs, settings.kpm_moments, settings.kpm_vectors)
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
    exact_max_orbitals=get_dos_exact_max_orbitals(conf), nelec=get_dos_nelec(conf))

"""
    compute_dos(Hr, Rs, orbitals, atoms, settings=DOSSettings(); ks=zeros(3, 1), soc=false)

Compute the DOS, PDOS and COHP of a real-space Hamiltonian.

# Arguments
- `Hr`: Real-space Hamiltonian, one matrix per lattice vector (as returned by `get_hr`).
- `Rs`: Lattice vectors (in units of the lattice vectors), one per column.
- `orbitals`: Name of every orbital, e.g. `"Pb-s"` or `"Br-px↑"` (see `orbital_labels`).
- `atoms`: Atom index of every orbital (used to exclude on-site terms from the COHP).
- `settings`: `DOSSettings`.
- `ks`: k-points (in units of the reciprocal lattice vectors) with equal weights.
- `soc`: Whether the orbitals are spin orbitals (spin degeneracy 1 instead of 2).

# Returns
A named tuple with `energies`, `dos`, `pdos` (energies × labels), `cohp` (energies × pairs), `labels`, `pairs`, `method`,
and, if `settings.nelec` is set, `fermi_energy`, `idos`, `icohp` (integrated curves) and `icohp_fermi` (ICOHP at the Fermi
energy).
"""
function compute_dos(Hr, Rs, orbitals, atoms, settings=DOSSettings(); ks=zeros(3, 1), soc=false)
    labels = unique(projection_label.(orbitals, settings.projection))
    label_index = [findfirst(==(projection_label(o, settings.projection)), labels) for o in orbitals]
    members = [findall(==(l), label_index) for l in eachindex(labels)]
    pairs = "cohp" ∈ settings.quantities ? cohp_pairs(Hr, atoms, label_index, labels, settings.cohp_pairs) : Tuple{Int64, Int64}[]
    g = soc ? 1 : 2
    method = settings.method == "auto" ? (length(orbitals) ≤ settings.exact_max_orbitals ? "exact" : "kpm") : settings.method
    result = if method == "exact"
        dos_exact(Hr, Rs, ks, atoms, members, pairs, g, settings)
    elseif method == "kpm"
        dos_kpm(Hr, Rs, ks, atoms, members, pairs, g, settings)
    else
        error("Unknown DOS method '$method'. Valid options are auto, exact and kpm.")
    end
    integrated = isnan(settings.nelec) ? (;) :
        (idos=cumulative_integral(result.energies, result.dos), icohp=reduce(hcat, [cumulative_integral(result.energies, c) for c in eachcol(result.cohp)], init=zeros(length(result.energies), 0)))
    return merge(result, integrated, (labels=labels, pairs=["$(labels[a])-$(labels[b])" for (a, b) in pairs], method=method))
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
    run_dos(ham, strcs, bases, local_inds, comm, conf=get_empty_config(); rank=0, nranks=1, ks=get_kpoints_from_config(conf))

Post-processing step of a standard run: compute the DOS, PDOS and COHP of every structure from its real-space Hamiltonian
and write them to the file `filename` of the `DOS` block (default `dos.h5`), one group per structure.
"""
function run_dos(ham, strcs, bases, local_inds, comm, conf=get_empty_config(); rank=0, nranks=1, ks=get_kpoints_from_config(conf),
                 verbosity=get_verbosity(conf))
    settings = DOSSettings(conf)
    filename = get_dos_filename(conf)
    soc = get_soc(conf)
    results = map(eachindex(strcs)) do index
        system, config_index = get_system_and_config_index(index, local_inds)
        orbitals, atoms = orbital_labels(strcs[index], bases[index], soc=soc)
        time = @elapsed result = compute_dos(get_hr(ham, index), ham.Rs[index], orbitals, atoms, settings, ks=ks, soc=soc)
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
