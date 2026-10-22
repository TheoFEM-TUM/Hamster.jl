"""
    MDRepulsive

Nuclear term of the MD energy, linear in its coefficients so that it is learned by linear least squares from the
difference between DFT and electronic energies and forces:

    E = α E_el + Σ_s e_s N_s + Σ_(i<j) φ_(s_i s_j)(r_ij) + Σ_i Σ_(j<k) Σ_l c_(s_i;s_j s_k),l P_l(cos θ_jik) w(r_ij) w(r_ik)
        + λ_C E_C + E_wall

- `α`: scale of the electronic free energy E_el (if `electron_scale`, else 1): a TB model fitted to band structures does
  not necessarily give accurate forces, which the fit can weight down;
- `e_s`: energy per atom of species `s` (reference energies of the DFT data);
- `φ_AB(r) = Σ_k c_ABk B_k(r)`: pair potential of the species pair (A, B), cubic B-splines with the knot spacing
  `h = (rcut - rmin) / nknots` that vanish smoothly (C²) at `rcut`;
- three-body terms of the angle θ_jik at atom i between its neighbors j and k (closer than `rcut3`): Legendre
  polynomials `P_l` up to `lmax3` (−1: none) times `w(r) = (1 - r/rcut3)³`, for every species triplet (center; pair);
- `E_C`: Coulomb energy of the effective charges `qeff` (Wolf/Zahn sum of the `Ewald` block) with the scale `λ_C`;
- `E_wall = Σ A (rmin - r)³` below `rmin` of each pair: a fixed repulsion where no training data constrain the splines.

# Fields
- `species`: types with an atomic energy; `pairs`: species pairs (sorted); `rmin`: start of the spline range per pair.
- `triplets`: species triplets (center, a ≤ b) of the three-body terms.
- `coeffs`: `[e_s..., c_AB1..c_ABK for every pair..., c_T0..c_Tlmax3 for every triplet..., λ_C, α]` (λ_C and α only if
  `coulomb` and `electron_scale`).
"""
mutable struct MDRepulsive
    rcut :: Float64
    nknots :: Int
    species :: Vector{UInt8}
    pairs :: Vector{Tuple{UInt8, UInt8}}
    rmin :: Vector{Float64}
    wall :: Float64
    rcut3 :: Float64
    lmax3 :: Int
    triplets :: Vector{Tuple{UInt8, UInt8, UInt8}}
    coulomb :: Bool
    electron_scale :: Bool
    coeffs :: Vector{Float64}
end

nthreebody(rep::MDRepulsive) = rep.lmax3 ≥ 0 ? length(rep.triplets) * (rep.lmax3 + 1) : 0
ncoeffs(rep::MDRepulsive) = length(rep.species) + length(rep.pairs) * rep.nknots + nthreebody(rep) + rep.coulomb + rep.electron_scale
spline_offset(rep::MDRepulsive, P) = length(rep.species) + (P - 1) * rep.nknots
threebody_offset(rep::MDRepulsive, T) = length(rep.species) + length(rep.pairs) * rep.nknots + (T - 1) * (rep.lmax3 + 1)
coulomb_index(rep::MDRepulsive) = length(rep.species) + length(rep.pairs) * rep.nknots + nthreebody(rep) + 1
electron_index(rep::MDRepulsive) = ncoeffs(rep)

"""
    electron_scale(rep)

Scale α of the electronic free energy in the MD energy (1 without `electron_scale`).
"""
electron_scale(rep::MDRepulsive) = rep.electron_scale ? rep.coeffs[electron_index(rep)] : 1.0
electron_scale(::Nothing) = 1.0

function MDRepulsive(species, rmin_pairs::Dict, conf=get_empty_config(); coulomb=get_md_rep_coulomb(conf),
                     electron_scale=get_md_rep_electron_scale(conf), lmax3=get_md_rep_lmax3(conf))
    species = sort(unique(species))
    pairs = [(a, b) for (i, a) in enumerate(species) for b in species[i:end]]
    triplets = [(c, a, b) for c in species for (a, b) in pairs]
    rmin = [rmin_pairs[P] for P in pairs]
    rep = MDRepulsive(get_md_rep_rcut(conf), get_md_rep_nknots(conf), species, pairs, rmin, get_md_rep_wall(conf),
                      get_md_rep_rcut3(conf), lmax3, triplets, coulomb, electron_scale, Float64[])
    rep.coeffs = zeros(ncoeffs(rep))
    electron_scale && (rep.coeffs[electron_index(rep)] = 1.0)
    return rep
end

triplet_index(rep::MDRepulsive, c, a, b) = findfirst(==((c, min(a, b), max(a, b))), rep.triplets)

pair_index(rep::MDRepulsive, a, b) = findfirst(==(a ≤ b ? (a, b) : (b, a)), rep.pairs)

"""
    bspline3(x)

Cardinal cubic B-spline on [0, 4] and its derivative.
"""
@inline function bspline3(x)
    if x ≤ 0 || x ≥ 4
        return 0.0, 0.0
    elseif x < 1
        return x^3 / 6, x^2 / 2
    elseif x < 2
        return (-3x^3 + 12x^2 - 12x + 4) / 6, (-9x^2 + 24x - 12) / 6
    elseif x < 3
        return (3x^3 - 24x^2 + 60x - 44) / 6, (9x^2 - 48x + 60) / 6
    else
        return (4 - x)^3 / 6, -(4 - x)^2 / 2
    end
end

"""
    spline_basis(rep, P, r)

Index of the first nonzero B-spline of pair `P` at distance `r` and the values and derivatives of the (at most four)
nonzero ones, `(k0, vals, ders)` with B-spline `k0 + m` for `m = 0:3` (indices outside 1:nknots are zero).
"""
@inline function spline_basis(rep::MDRepulsive, P, r)
    K = rep.nknots
    h = (rep.rcut - rep.rmin[P]) / K
    # Knots t_k = rcut - (K - k + 4) h, support of B_k: [t_k, t_k + 4h]
    t1 = rep.rcut - (K + 3) * h
    x = (r - t1) / h                  # position relative to t_1 in units of h
    k0 = floor(Int, x) - 2            # B-splines k0 .. k0+3 can be nonzero
    vals = MVector{4, Float64}(0, 0, 0, 0); ders = MVector{4, Float64}(0, 0, 0, 0)
    for m in 0:3
        k = k0 + m
        (1 ≤ k ≤ K) || continue
        v, dv = bspline3(x - (k - 1))
        vals[m+1] = v; ders[m+1] = dv / h
    end
    return k0, vals, ders
end

"""
    pair_energy(rep, P, r)

Pair potential (splines and wall, without Coulomb) of pair `P` at `r` and its derivative.
"""
function pair_energy(rep::MDRepulsive, P, r)
    r ≥ rep.rcut && return 0.0, 0.0
    k0, vals, ders = spline_basis(rep, P, r)
    off = spline_offset(rep, P)
    e = 0.0; de = 0.0
    for m in 0:3
        k = k0 + m
        (1 ≤ k ≤ rep.nknots) || continue
        e += rep.coeffs[off+k] * vals[m+1]; de += rep.coeffs[off+k] * ders[m+1]
    end
    ew, dew = wall_energy(rep, P, r)
    return e + ew, de + dew
end

@inline function wall_energy(rep::MDRepulsive, P, r)
    x = rep.rmin[P] - r
    x > 0 || return 0.0, 0.0
    return rep.wall * x^3, -3rep.wall * x^2
end

"""
    legendre!(P, dP, x)

Legendre polynomials `P[l+1] = P_l(x)` and their derivatives for `l = 0:length(P)-1`.
"""
function legendre!(P, dP, x)
    L = length(P) - 1
    P[1] = 1.0; dP[1] = 0.0
    L ≥ 1 && (P[2] = x; dP[2] = 1.0)
    for l in 1:L-1
        P[l+2] = ((2l + 1) * x * P[l+1] - l * P[l]) / (l + 1)
        dP[l+2] = dP[l] + (2l + 1) * P[l+1]
    end
    return nothing
end

"""
    threebody_center!(f, rep, nl, positions, L, types, i, P, dP)

Three-body terms of the center atom `i`: for all pairs of its neighbors j, k (closer than `rep.rcut3`) and all orders
l, calls `f(T, l, g, j, k, a, b, ∂g/∂a, ∂g/∂b)` with the triplet `T`, the value `g = P_l(cos θ) w(|a|) w(|b|)` and its
derivatives with respect to the bond vectors `a = r_j - r_i` and `b = r_k - r_i`.
"""
function threebody_center!(f, rep::MDRepulsive, nl, positions, L, types, i, P, dP)
    rc = rep.rcut3
    ps = nl.first[i]:nl.first[i+1]-1
    for (ip, p) in enumerate(ps)
        a = bond_vector(positions, L, i, nl.j[p], nl.n[p])
        ra = norm(a)
        ra < rc || continue
        wa = (1 - ra / rc)^3; dwa = -3 / rc * (1 - ra / rc)^2
        for q in ps[ip+1:end]
            b = bond_vector(positions, L, i, nl.j[q], nl.n[q])
            rb = norm(b)
            rb < rc || continue
            T = triplet_index(rep, types[i], types[nl.j[p]], types[nl.j[q]])
            isnothing(T) && continue
            wb = (1 - rb / rc)^3; dwb = -3 / rc * (1 - rb / rc)^2
            c = clamp(dot(a, b) / (ra * rb), -1.0, 1.0)
            dca = b / (ra * rb) - c * a / ra^2
            dcb = a / (ra * rb) - c * b / rb^2
            legendre!(P, dP, c)
            for l in 0:rep.lmax3
                g = P[l+1] * wa * wb
                ga = (dP[l+1] * wa * wb) * dca + (P[l+1] * dwa * wb / ra) * a
                gb = (dP[l+1] * wa * wb) * dcb + (P[l+1] * wa * dwb / rb) * b
                f(T, l, g, nl.j[p], nl.j[q], a, b, ga, gb)
            end
        end
    end
    return nothing
end

"""
    threebody_energy_forces!(F, rep, nl, positions, L, types)

Energy of the three-body terms, their forces (added to `F`) and virial.
"""
function threebody_energy_forces!(F, rep::MDRepulsive, nl, positions, L, types)
    nthreebody(rep) == 0 && return 0.0, zero(SMatrix{3, 3, Float64})
    N = length(types)
    parts = tmap(chunks(1:N, n=4 * Threads.nthreads())) do atoms
        Floc = zeros(3, N)
        P = zeros(rep.lmax3 + 1); dP = zeros(rep.lmax3 + 1)
        Eloc = 0.0; Wloc = zero(SMatrix{3, 3, Float64})
        for i in atoms
            threebody_center!(rep, nl, positions, L, types, i, P, dP) do T, l, g, j, k, a, b, ga, gb
                c = rep.coeffs[threebody_offset(rep, T) + l + 1]
                Eloc += c * g
                for x in 1:3
                    Floc[x, j] -= c * ga[x]; Floc[x, k] -= c * gb[x]; Floc[x, i] += c * (ga[x] + gb[x])
                end
                Wloc += c * (a * ga' + b * gb')
            end
        end
        (Eloc, Wloc, Floc)
    end
    for part in parts
        F .+= part[3]
    end
    return sum(p -> p[1], parts), sum(p -> p[2], parts)
end

"""
    repulsive_energy_forces!(F, rep, nl, positions, L, types; wolf=nothing)

Energy of the nuclear term, its forces (added to `F`) and virial, for the neighbor list `nl` (cutoff ≥ `rep.rcut`, and
≥ the Wolf cutoff if the Coulomb term is used with the Wolf model `wolf`). With `nl_coulomb`, the Coulomb term uses its
own list.
"""
function repulsive_energy_forces!(F, rep::MDRepulsive, nl, positions, L, types; wolf=nothing, nl_coulomb=nl)
    N = length(types)
    pidx = Dict((a, b) => pair_index(rep, a, b) for a in unique(types) for b in unique(types))
    parts = tmap(chunks(1:N, n=4 * Threads.nthreads())) do atoms
        E = 0.0; W = zero(SMatrix{3, 3, Float64})
        for a in atoms
            Fa = zero(SVector{3, Float64})
            for p in nl.first[a]:nl.first[a+1]-1
                b = nl.j[p]
                P = pidx[(types[a], types[b])]
                isnothing(P) && continue
                d = bond_vector(positions, L, a, b, nl.n[p])
                r = norm(d)
                r < rep.rcut || continue
                e, de = pair_energy(rep, P, r)
                E += e / 2
                Fa += (de / r) * d
                W += (de / 2r) * (d * d')
            end
            F[1, a] += Fa[1]; F[2, a] += Fa[2]; F[3, a] += Fa[3]
        end
        (E, W)
    end
    # (E and W are local to the tasks above; the totals have other names, which keeps them local there)
    Etot = sum(first, parts; init=0.0); Wtot = sum(last, parts; init=zero(SMatrix{3, 3, Float64}))
    E3, W3 = threebody_energy_forces!(F, rep, nl, positions, L, types)
    Etot += E3; Wtot += W3
    for (s, t) in enumerate(rep.species)
        Etot += rep.coeffs[s] * count(==(t), types)
    end
    if rep.coulomb && !isnothing(wolf)
        λ = rep.coeffs[coulomb_index(rep)]
        if λ ≠ 0
            Etot += λ * coulomb_energy(wolf, nl_coulomb, positions, L)
            Wtot += pair_kernel_forces!(F, wolf, nl_coulomb, positions, L, (a, b) -> λ * wolf.q[a] * wolf.q[b] / 2)
        end
    end
    return Etot, Wtot
end

"""
    coulomb_energy(wolf, nl, positions, L)

Coulomb energy `1/2 Σ_(i,j) ke q_i q_j K(r_ij)` of the effective charges with the Wolf/Zahn kernel.
"""
function coulomb_energy(wolf::MDWolf, nl, positions, L)
    return tmapreduce(+, 1:length(nl.first)-1) do a
        e = 0.0
        for p in nl.first[a]:nl.first[a+1]-1
            r = norm(bond_vector(positions, L, a, nl.j[p], nl.n[p]))
            r < wolf.rc && (e += wolf.q[a] * wolf.q[nl.j[p]] * wolf.kernel(r, wolf.α, wolf.rc))
        end
        ke * e / 2
    end
end

"""
    repulsive_features(rep, nl, positions, L, types; wolf=nothing, nl_coulomb=nl)

Design rows of one structure for the linear fit: the energy features `xE` (`E_rep = xE ⋅ coeffs + E_wall`) and the force
features `XF` (3N × ncoeffs, `F_rep = XF coeffs + F_wall`), with the energy and forces of the fixed wall. The columns of
the electronic scale are left zero (filled by the fit with the electronic energies and forces).
"""
function repulsive_features(rep::MDRepulsive, nl, positions, L, types; wolf=nothing, nl_coulomb=nl)
    N = length(types)
    nc = ncoeffs(rep)
    xE = zeros(nc); XF = zeros(3N, nc)
    Ewall = 0.0; Fwall = zeros(3, N)
    for (s, t) in enumerate(rep.species)
        xE[s] = count(==(t), types)
    end
    pidx = Dict((a, b) => pair_index(rep, a, b) for a in unique(types) for b in unique(types))
    for a in 1:N, p in nl.first[a]:nl.first[a+1]-1
        b = nl.j[p]
        P = pidx[(types[a], types[b])]
        isnothing(P) && continue
        d = bond_vector(positions, L, a, b, nl.n[p])
        r = norm(d)
        r < rep.rcut || continue
        k0, vals, ders = spline_basis(rep, P, r)
        off = spline_offset(rep, P)
        for m in 0:3
            k = k0 + m
            (1 ≤ k ≤ rep.nknots) || continue
            xE[off+k] += vals[m+1] / 2
            for x in 1:3
                XF[3(a-1)+x, off+k] += ders[m+1] * d[x] / r
            end
        end
        ew, dew = wall_energy(rep, P, r)
        Ewall += ew / 2
        Fwall[:, a] .+= (dew / r) .* d
    end
    if nthreebody(rep) > 0
        P = zeros(rep.lmax3 + 1); dP = zeros(rep.lmax3 + 1)
        for i in 1:N
            threebody_center!(rep, nl, positions, L, types, i, P, dP) do T, l, g, j, k, a, b, ga, gb
                col = threebody_offset(rep, T) + l + 1
                xE[col] += g
                for x in 1:3
                    XF[3(j-1)+x, col] -= ga[x]; XF[3(k-1)+x, col] -= gb[x]; XF[3(i-1)+x, col] += ga[x] + gb[x]
                end
            end
        end
    end
    if rep.coulomb && !isnothing(wolf)
        c = coulomb_index(rep)
        xE[c] = coulomb_energy(wolf, nl_coulomb, positions, L)
        Fc = zeros(3, N)
        pair_kernel_forces!(Fc, wolf, nl_coulomb, positions, L, (a, b) -> wolf.q[a] * wolf.q[b] / 2)
        XF[:, c] .= vec(Fc)
    end
    return xE, XF, Ewall, Fwall
end

"""
    save_repulsive(rep, filename; meta=Dict())

Write the nuclear term (and settings in `meta`, e.g. of the electrons it was fitted with) to the HDF5 file `filename`.
"""
function save_repulsive(rep::MDRepulsive, filename; meta=Dict{String, Any}())
    h5open(filename, "w") do f
        f["rcut"] = rep.rcut; f["nknots"] = rep.nknots
        f["species"] = Int.(rep.species)
        f["pairs"] = [Int(P[k]) for k in 1:2, P in rep.pairs]
        f["rmin"] = rep.rmin; f["wall"] = rep.wall; f["coulomb"] = rep.coulomb; f["coeffs"] = rep.coeffs
        f["rcut3"] = rep.rcut3; f["lmax3"] = rep.lmax3; f["electron_scale"] = rep.electron_scale
        f["triplets"] = [Int(T[k]) for k in 1:3, T in rep.triplets]
        for (key, value) in meta
            f["meta/$key"] = value
        end
    end
end

"""
    load_repulsive(filename)

Read a nuclear term written by `save_repulsive`.
"""
function load_repulsive(filename)
    isfile(filename) || error("Nuclear term $filename of the MD not found: fit it first (task = fit).")
    h5open(filename, "r") do f
        pairs = [(UInt8(c[1]), UInt8(c[2])) for c in eachcol(read(f["pairs"]))]
        triplets = [(UInt8(c[1]), UInt8(c[2]), UInt8(c[3])) for c in eachcol(read(f["triplets"]))]
        return MDRepulsive(read(f["rcut"]), read(f["nknots"]), UInt8.(read(f["species"])), pairs, read(f["rmin"]),
                           read(f["wall"]), read(f["rcut3"]), read(f["lmax3"]), triplets, read(f["coulomb"]),
                           read(f["electron_scale"]), read(f["coeffs"]))
    end
end

"""
    write_repulsive_table(rep, filename; npoints=500)

Tabulate the pair potentials (without Coulomb) of `rep` in a text file (columns r and one per pair).
"""
function write_repulsive_table(rep::MDRepulsive, filename; npoints=500)
    rs = range(max(0.5, minimum(rep.rmin) - 0.5), rep.rcut, length=npoints)
    open(filename, "w") do io
        println(io, "# r(Å) ", join(["$(number_to_element(a))-$(number_to_element(b))(eV)" for (a, b) in rep.pairs], " "))
        for r in rs
            @printf(io, "%10.5f", r)
            for P in eachindex(rep.pairs)
                @printf(io, " %14.8f", pair_energy(rep, P, r)[1])
            end
            println(io)
        end
    end
end
