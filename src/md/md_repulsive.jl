"""
    MDRepulsive

Nuclear (repulsive) term of the MD energy, linear in its coefficients so that it is learned by linear least squares
from the difference between DFT and electronic energies and forces:

    E_rep = Σ_s e_s N_s + Σ_(i<j) φ_(s_i s_j)(r_ij) + λ_C E_C + E_wall

- `e_s`: energy per atom of species `s` (reference energies of the DFT data);
- `φ_AB(r) = Σ_k c_ABk B_k(r)`: pair potential of the species pair (A, B), cubic B-splines with the knot spacing
  `h = (rcut - rmin) / nknots` that vanish smoothly (C²) at `rcut`;
- `E_C`: Coulomb energy of the effective charges `qeff` (Wolf/Zahn sum of the `Ewald` block) with the scale `λ_C`;
- `E_wall = Σ A (rmin - r)³` below `rmin` of each pair: a fixed repulsion where no training data constrain the splines.

# Fields
- `species`: types with an atomic energy; `pairs`: species pairs (sorted); `rmin`: start of the spline range per pair.
- `coeffs`: `[e_s..., c_AB1..c_ABK for every pair..., λ_C]`.
"""
mutable struct MDRepulsive
    rcut :: Float64
    nknots :: Int
    species :: Vector{UInt8}
    pairs :: Vector{Tuple{UInt8, UInt8}}
    rmin :: Vector{Float64}
    wall :: Float64
    coulomb :: Bool
    coeffs :: Vector{Float64}
end

ncoeffs(rep::MDRepulsive) = length(rep.species) + length(rep.pairs) * rep.nknots + (rep.coulomb ? 1 : 0)
spline_offset(rep::MDRepulsive, P) = length(rep.species) + (P - 1) * rep.nknots
coulomb_index(rep::MDRepulsive) = ncoeffs(rep)

function MDRepulsive(species, rmin_pairs::Dict, conf=get_empty_config(); coulomb=get_md_rep_coulomb(conf))
    species = sort(unique(species))
    pairs = [(a, b) for (i, a) in enumerate(species) for b in species[i:end]]
    rmin = [rmin_pairs[P] for P in pairs]
    rep = MDRepulsive(get_md_rep_rcut(conf), get_md_rep_nknots(conf), species, pairs, rmin, get_md_rep_wall(conf), coulomb, Float64[])
    rep.coeffs = zeros(ncoeffs(rep))
    return rep
end

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
features `XF` (3N × ncoeffs, `F_rep = XF coeffs + F_wall`), with the energy and forces of the fixed wall.
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
        return MDRepulsive(read(f["rcut"]), read(f["nknots"]), UInt8.(read(f["species"])), pairs, read(f["rmin"]),
                           read(f["wall"]), read(f["coulomb"]), read(f["coeffs"]))
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
