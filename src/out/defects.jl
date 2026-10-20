# Regions around the defects of a supercell (found by the mapping onto a primitive cell, see UnfoldMap): the defect
# atoms (substituted and unmapped atoms), their neighbour shells and the remaining (bulk) atoms. Used for the
# defect-projected DOS (projection = defect).

"""
    periodic_distance(lattice, r1, r2)

Shortest distance (Å) between the Cartesian positions `r1` and `r2` and their periodic images in the cell `lattice`.
"""
function periodic_distance(lattice, r1, r2)
    f = lattice \ (r1 - r2)
    f = f .- round.(f)
    return minimum(norm(lattice * (f + [i, j, k])) for i in -1:1, j in -1:1, k in -1:1)
end

"""
    site_neighbor_distances(pc_lattice, pc_positions)

Nearest-neighbour distance (Å) of every site of the primitive cell (fractional `pc_positions`, columns).
"""
function site_neighbor_distances(pc_lattice, pc_positions)
    cart = pc_lattice * pc_positions
    return map(axes(cart, 2)) do q
        minimum(norm(cart[:, p] + pc_lattice * [i, j, k] - cart[:, q])
                for p in axes(cart, 2), i in -2:2, j in -2:2, k in -2:2 if p ≠ q || (i, j, k) ≠ (0, 0, 0))
    end
end

"""
    neighbor_shells(lattice, positions, radius, core, centers, shells; factor=1.2, separate=true, core_defects=nothing) -> (levels, owner)

Shells around the `core` atoms and the additional `centers` (positions and radii, e.g. vacant sites): `levels` is 0
for the core atoms, n = 1, …, `shells` for the n-th neighbour shell, -1 for the remaining atoms. An atom is a neighbour
of an atom or center of the previous shell if their distance is below `factor` times the larger of their `radius`
(nearest-neighbour distances).

With `separate`, core atoms and centers that are neighbours belong to the same defect (e.g. a split interstitial or a
Frenkel pair), and `owner` is the defect (1, 2, …) of every atom: of its core atom, or for a shell atom, of the closest
atom or center of the previous shell (0 for the remaining atoms). Without `separate`, all defects are one (owner 1).
`core_defects` gives the defect of every core atom directly instead (e.g. the localized state it belongs to; no centers).
"""
function neighbor_shells(lattice, positions, radius, core, centers, shells; factor=1.2, separate=true, core_defects=nothing)
    nodes = [[(positions[a], radius[a]) for a in core]; centers]
    neighbors(x, y) = periodic_distance(lattice, x[1], y[1]) < factor * max(x[2], y[2])
    # Defects: connected groups of core atoms and centers
    label = collect(eachindex(nodes))
    root(i) = label[i] == i ? i : root(label[i])
    if separate && isnothing(core_defects)
        for i in eachindex(nodes), j in 1:i-1
            neighbors(nodes[i], nodes[j]) && (label[root(i)] = root(j))
        end
    end
    roots = unique(root.(eachindex(nodes)))
    defect = !separate ? ones(Int64, length(nodes)) : !isnothing(core_defects) ? collect(core_defects) :
             [findfirst(==(root(i)), roots) for i in eachindex(nodes)]
    levels = fill(-1, length(positions)); owner = zeros(Int64, length(positions))
    levels[core] .= 0; owner[core] .= defect[eachindex(core)]
    frontier = [(nodes[i]..., defect[i]) for i in eachindex(nodes)]
    for shell in 1:shells
        new = Int64[]
        for a in eachindex(positions)
            levels[a] == -1 || continue
            d = [periodic_distance(lattice, positions[a], r) for (r, _, _) in frontier]
            close = findall(n -> d[n] < factor * max(frontier[n][2], radius[a]), eachindex(frontier))
            isempty(close) && continue
            push!(new, a); owner[a] = frontier[close[argmin(d[close])]][3]
        end
        isempty(new) && break
        levels[new] .= shell
        frontier = [(positions[a], radius[a], owner[a]) for a in new]
    end
    return levels, owner
end

"""
    nearest_neighbor_distances(lattice, positions)

Distance (Å) from every atom to its closest neighbour (periodic images included).
"""
nearest_neighbor_distances(lattice, positions) =
    [minimum((periodic_distance(lattice, positions[a], positions[b]) for b in eachindex(positions) if b ≠ a), init=Inf)
     for a in eachindex(positions)]

"""
    defect_regions(strc, umap, pc_lattice, pc_positions; shells=2, displaced=false, factor=1.2, separate=true) -> (levels, owner)

Region of every atom of `strc` around the defects of the mapping `umap`: 0 for the defect atoms (substituted and
unmapped atoms), n = 1, …, `shells` for the n-th neighbour shell, -1 for the bulk. The shells are built from the defect
atoms and the vacant sites outwards: an atom is a neighbour of an atom or vacant site of the previous shell if their
distance is below `factor` times the larger of their nearest-neighbour distances (that of the primitive site for mapped
atoms and vacancies, the distance to the closest atom for unmapped atoms). The positions are the reference positions,
or with `displaced` the positions including the distortions (as in the mapping). `owner` is the defect of every atom
(see `neighbor_shells`): separate defects (e.g. two vacancies far apart) get separate regions.
"""
function defect_regions(strc, umap, pc_lattice, pc_positions; shells=2, displaced=false, factor=1.2, separate=true)
    positions = [displaced ? ion.pos + ion.dist : ion.pos for ion in strc.ions]
    dsite = site_neighbor_distances(pc_lattice, pc_positions)
    own = nearest_neighbor_distances(strc.lattice, positions)
    radius = [umap.site[a] > 0 ? dsite[umap.site[a]] : own[a] for a in eachindex(positions)]
    core = sort(unique([umap.substituted; umap.unmapped]))
    vacancies = [(umap.vacant_positions[:, v], dsite[umap.vacant[v]]) for v in eachindex(umap.vacant)]
    return neighbor_shells(strc.lattice, positions, radius, core, vacancies, shells, factor=factor, separate=separate)
end

# ==============================================================================================================
# Localized states
# ==============================================================================================================

"""
    participation(energies, states, atom_types, σ, wk) -> (pn, ratio, wa, density)

Energy-resolved participation of the atoms in the states `states` (per k-point: eigenvalues `E` and atom weights `Wa`,
atoms × states, Σₐ Wa = 1), with Gaussian broadening `σ` and k-point weight `wk`. The atom weights of the states near
E are averaged, wₐ(E) = Σₙ g(E - Eₙ) Waₙ / Σₙ g(E - Eₙ), which does not depend on the choice of eigenvectors within
(nearly) degenerate states, and

- `pn[e]` = 1 / Σₐ wₐ(E)²: the number of atoms the states at E live on;
- `ratio[e]` = pn / pn_uniform, where pn_uniform = 1 / Σ_X W_X² / n_X is the number for states with the same weights
  W_X on the elements X (n_X atoms) spread uniformly over the atoms of every element: about 1 for Bloch-like states
  (also on a single sublattice), small for localized states;
- `wa` (atoms × energies): the atom weights wₐ(E);
- `density`: Σₙ g(E - Eₙ) wk (the DOS per spin).

Energies without states (Σₙ g < 1e-8 / eV) have `pn = ratio = NaN`.
"""
function participation(energies, states, atom_types, σ, wk)
    natoms = length(atom_types)
    num = zeros(natoms, length(energies)); den = zeros(length(energies))
    for (E, Wa) in states
        G = exp.(-0.5 .* ((energies .- transpose(E)) ./ σ) .^ 2) ./ (σ * sqrt(2π)) .* wk   # energies × states
        num .+= Wa * transpose(G)
        den .+= vec(sum(G, dims=2))
    end
    elements = unique(atom_types)
    members = [findall(==(X), atom_types) for X in elements]
    pn = fill(NaN, length(energies)); ratio = fill(NaN, length(energies))
    wa = zeros(natoms, length(energies))
    for e in eachindex(energies)
        den[e] < 1e-8 && continue
        w = view(num, :, e) ./ sum(view(num, :, e))
        wa[:, e] .= w
        pn[e] = 1 / sum(abs2, w)
        uniform = 1 / sum(sum(w[m])^2 / length(m) for m in members)
        ratio[e] = pn[e] / uniform
    end
    return pn, ratio, wa, den
end

"""
    atom_weight_states(Hr, Rs, ks, atoms)

Diagonalize the Hamiltonian at every k-point and return, per k-point, the eigenvalues and the weights of the atoms in
every state (atoms × states, from the orbital `atoms`).
"""
function atom_weight_states(Hr, Rs, ks, atoms)
    members = [findall(==(a), atoms) for a in 1:maximum(atoms)]
    return tmap(axes(ks, 2)) do k
        E, V = eigen(Hermitian(Matrix(get_hamiltonian(Hr, Rs, ks[:, k:k])[1])))
        absV = abs2.(V)
        (E, reduce(vcat, [sum(absV[m, :], dims=1) for m in members]))
    end
end

"""
    localized_atoms(energies, ratio, pn, wa, dos; threshold=0.1, emin=-Inf, emax=Inf, dos_min=1e-3) -> (atoms, peaks)

Atoms that carry the localized states: for every energy interval in [`emin`, `emax`] where the participation ratio (see
`participation`) is below `threshold` and the DOS is above `dos_min` times its maximum, the `round(pn)` atoms with the
largest weights at the maximum of the DOS in the interval (the participation number is the effective number of atoms
of the states). Returns the union of these atoms and the intervals (named tuples with the energy, participation ratio
and number at this maximum and the atoms).
"""
function localized_atoms(energies, ratio, pn, wa, dos; threshold=0.1, emin=-Inf, emax=Inf, dos_min=1e-3)
    inside = [emin ≤ energies[e] ≤ emax && !isnan(ratio[e]) && ratio[e] < threshold && dos[e] > dos_min * maximum(dos)
              for e in eachindex(energies)]
    peaks = NamedTuple[]
    e = 1
    while e ≤ length(energies)
        if inside[e]
            f = e
            while f < length(energies) && inside[f+1]; f += 1; end
            best = e - 1 + argmax(dos[e:f])      # the energy of the localized states
            order = sortperm(wa[:, best], rev=true)
            push!(peaks, (energy=energies[best], ratio=ratio[best], pn=pn[best], atoms=sort(order[1:max(1, round(Int64, pn[best]))])))
            e = f + 1
        else
            e += 1
        end
    end
    return sort(unique(reduce(vcat, (p.atoms for p in peaks), init=Int64[]))), peaks
end

"""
    region_name(level, core="defect", owner=1, ndefects=1)

Name of a region: `core` (e.g. `"defect"`), `"shell<n>"` or `"bulk"`; with several defects, numbered by the defect,
e.g. `"defect2"` and `"defect2-shell1"`.
"""
function region_name(level, core="defect", owner=1, ndefects=1)
    level < 0 && return "bulk"
    ndefects == 1 && return level == 0 ? core : "shell$level"
    return level == 0 ? "$core$owner" : "$core$owner-shell$level"
end
