"""
    MDWolf

Electrostatic onsite corrections of the `Ewald` block for the MD (Wolf or Zahn sum, which are pair sums and therefore
have simple forces): `Δε_i = -γ_s (φ_i - φ̄_s)` on all orbitals of atom `i` of species `s`, with the potential
`φ_i = ke Σ_j q_j K(r_ij)` of the effective charges `qeff` and the mean `φ̄_s` of species `s` in the structure (as for a
single structure in Hamster).

# Fields
- `kernel`: `wolf_kernel` or `zahn_kernel`; `α`, `rc`: damping and cutoff.
- `q`, `γ`: charge and coupling of every atom; `species`: index of the species of every atom (0 without orbitals).
- `nspecies`: number of species with orbitals.
"""
struct MDWolf{K}
    kernel :: K
    method :: String
    α :: Float64
    rc :: Float64
    q :: Vector{Float64}
    γ :: Vector{Float64}
    species :: Vector{Int}
    nspecies :: Int
end

"""
    MDWolf(types, norb, conf)

Wolf/Zahn onsite model for atoms of `types` (with `norb` orbitals) with the settings of the `Ewald` block (method,
rcut, alpha, qeff of the atom blocks, couplings from `init_params`). `method = ewald` is not supported (no forces of the
reciprocal-space sum), use wolf or zahn.
"""
function MDWolf(types, norb, conf=get_empty_config())
    method = lowercase(get_ewald_method(conf))
    method ∈ ("wolf", "zahn") || error("The MD supports the Ewald block only with method = wolf or zahn (not $method).")
    rc = get_ewald_rcut(conf)
    α = ewald_alpha(method, rc, get_ewald_alpha(conf))
    q = [get_qeff(conf, number_to_element(t)) for t in types]
    abs(sum(q)) < 1e-6 || error("The effective charges qeff of the MD cell sum to $(sum(q)) ≠ 0.")
    labels = unique(types[norb .> 0])
    γlabels = init_ewald_params(labels, conf)
    γ = [norb[a] > 0 ? γlabels[findfirst(==(types[a]), labels)] : 0.0 for a in eachindex(types)]
    species = [norb[a] > 0 ? findfirst(==(types[a]), labels) : 0 for a in eachindex(types)]
    kernel = method == "wolf" ? wolf_kernel : zahn_kernel
    return MDWolf(kernel, method, α, rc, q, γ, species, length(labels))
end

"""
    kernel_derivative(kernel, r, α, rc)

Derivative of the real-space kernels `wolf_kernel` and `zahn_kernel` with respect to `r`.
"""
function kernel_derivative(kernel, r, α, rc)
    dK = -erfc(α * r) / r^2 - 2α / √π * exp(-(α * r)^2) / r
    kernel === zahn_kernel && (dK -= erfc(α * rc) / rc^2 + 2α / √π * exp(-(α * rc)^2) / rc)
    return dK
end

"""
    wolf_potentials!(φ, wolf, nl, positions, L)

Potentials `φ_i = ke Σ_j q_j K(r_ij)` of all atoms (pairs of the neighbor list `nl` within `wolf.rc`).
"""
function wolf_potentials!(φ, wolf::MDWolf, nl, positions, L)
    tforeach(chunks(1:length(nl.first)-1, n=4 * Threads.nthreads())) do atoms
        for a in atoms
            s = 0.0
            for p in nl.first[a]:nl.first[a+1]-1
                r = norm(bond_vector(positions, L, a, nl.j[p], nl.n[p]))
                r < wolf.rc && (s += wolf.q[nl.j[p]] * wolf.kernel(r, wolf.α, wolf.rc))
            end
            φ[a] = ke * s
        end
    end
    return φ
end

"""
    wolf_onsites(wolf, φ)

Onsite shifts `Δε_i = -γ_s (φ_i - φ̄_s)` of all atoms.
"""
function wolf_onsites(wolf::MDWolf, φ)
    means = species_means(wolf, φ)
    return [wolf.species[a] > 0 ? -wolf.γ[a] * (φ[a] - means[wolf.species[a]]) : 0.0 for a in eachindex(φ)]
end

function species_means(wolf::MDWolf, x)
    sums = zeros(wolf.nspecies); counts = zeros(wolf.nspecies)
    for a in eachindex(x)
        s = wolf.species[a]
        s > 0 || continue
        sums[s] += x[a]; counts[s] += 1
    end
    return sums ./ max.(counts, 1)
end

"""
    wolf_forces!(F, wolf, nl, positions, L, w)

Forces and virial of the onsite corrections for the electron numbers `w_i` of the atoms (spin included): the band
energy changes by `Σ_i w_i dΔε_i = Σ_i c_i dφ_i` with `c_i = -γ_s (w_i - w̄_s)` (the species mean is subtracted from
the potentials, so its derivative moves to the weights). Adds the forces to `F` and returns the virial `Σ d ⊗ ∂E/∂d`.
"""
function wolf_forces!(F, wolf::MDWolf, nl, positions, L, w)
    means = species_means(wolf, w)
    c = [wolf.species[a] > 0 ? -wolf.γ[a] * (w[a] - means[wolf.species[a]]) : 0.0 for a in eachindex(w)]
    return pair_kernel_forces!(F, wolf, nl, positions, L, (a, b) -> c[a] * wolf.q[b])
end

"""
    pair_kernel_forces!(F, wolf, nl, positions, L, coupling)

Forces of the energy `Σ_(i,j) coupling(i, j) ke K(r_ij)` over the directed pairs of `nl`, added to `F`; returns the
virial `Σ d ⊗ ∂E/∂d`. The list contains every pair in both directions, so the force on an atom is gathered from its own
pairs (`coupling(i, j) + coupling(j, i)`), without writes to other atoms.
"""
function pair_kernel_forces!(F, wolf::MDWolf, nl, positions, L, coupling)
    N = size(F, 2)
    Ws = tmap(chunks(1:N, n=4 * Threads.nthreads())) do atoms
        W = zero(SMatrix{3, 3, Float64})
        for a in atoms
            Fa = zero(SVector{3, Float64})
            for p in nl.first[a]:nl.first[a+1]-1
                b = nl.j[p]
                d = bond_vector(positions, L, a, b, nl.n[p])
                r = norm(d)
                r < wolf.rc || continue
                dK = ke * kernel_derivative(wolf.kernel, r, wolf.α, wolf.rc) / r
                Fa += (dK * (coupling(a, b) + coupling(b, a))) * d
                W += d * ((dK * coupling(a, b)) * d)'
            end
            F[1, a] += Fa[1]; F[2, a] += Fa[2]; F[3, a] += Fa[3]
        end
        W
    end
    return sum(Ws)
end
