"""
    SymmetryTerms

Variants of the descriptors of one structure (enforce_symmetry_group, see `symmetry_variants`): the ML correction of a
matrix element is ΔH_ij = Σ_v w_v K(x_v). Per lattice vector `R`: `base_weights[R][k]` is the weight of the k-th stored
descriptor (in the order of `findnz`), `extra[R]` holds the other variants as (index k, descriptor, weight).
"""
struct SymmetryTerms
    base_weights :: Vector{Vector{Float64}}
    extra :: Vector{Vector{Tuple{Int64, SVector{8, Float64}, Float64}}}
end

"""
    get_tb_descriptor_and_symmetry(h, V, strc, basis, conf; group=get_ml_enforce_symmetry_group(conf))

Descriptors of all matrix elements and, with a symmetry group, their `SymmetryTerms` (`nothing` for `"none"`, where the
descriptors are those of `get_tb_descriptor`).
"""
function get_tb_descriptor_and_symmetry(h, V, strc::Structure, basis, conf::Config; group=get_ml_enforce_symmetry_group(conf))
    group == "none" && return get_tb_descriptor(h, V, strc, basis, conf), nothing
    return symmetric_tb_descriptor(h, V, strc, basis, conf, group)
end

"""
    shell_rows(basis) -> Dict{Tuple{Int64, Int64}, Vector{Int64}}

Basis indices of the shells (ion, l) of pure orbitals with l ≥ 1, in the order of `shell_types(l)`. The shells must be
complete (all orbitals of the l), since the symmetry operations mix them.
"""
function shell_rows(basis)
    rows = Dict{Tuple{Int64, Int64}, Vector{Int64}}()
    i = 0
    for (iion, orbs) in enumerate(basis.orbitals), orb in orbs
        i += 1
        orb.type.l ≥ 1 || continue
        r = get!(rows, (iion, orb.type.l), zeros(Int64, 2orb.type.l + 1))
        r[shell_index(orb)] = i
    end
    for ((iion, l), r) in rows
        all(>(0), r) || error("enforce_symmetry_group needs complete shells: ion $iion has only some of its l = $l orbitals.")
    end
    return rows
end

"""
    shell_environment(h, strc, basis, rows, rcut) -> (env, E)

Squared environment descriptors: `env[i]` = Σ_k H_ik² f_cut(r_ik) over the orbitals k of the other atoms (geometry
tensor with unit parameters), and per shell (ion, l) the matrix `E[(ion, l)][a, b]` = Σ_k H_ak H_bk f_cut(r_ak), so that
an orbital with coefficients u in the shell has the environment uᵀ E u. A sum over complete shells of the neighbours,
it is invariant under rotations of the neighbouring orbitals and their sign conventions.
"""
function shell_environment(h, strc, basis, rows, rcut)
    Nε = length(basis); Norb = size(basis)
    Hr = get_hr(h, ones(size(h, 1)), Sparse())
    R0 = findR0(strc.Rs)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    rs_ion = get_ion_positions(strc.ions)
    index_map = get_index_to_ion_orb_map(Norb)
    # Columns of B_R: the weighted couplings √f H_ik of orbital i to all orbitals k of the other atoms
    Bs = map(enumerate(Hr)) do (R, H)
        I = Int64[]; J = Int64[]; vals = Float64[]
        for (i, k, Hval) in zip(findnz(H)...)
            iion, _ = index_map[i]; kion, _ = index_map[k]
            (iion ≠ kion || R ≠ R0) || continue
            f = fcut(normdiff(rs_ion[iion], rs_ion[kion], Ts[:, R]), rcut)
            f > 0 || continue
            push!(I, k); push!(J, i); push!(vals, real(Hval) * √f)
        end
        sparse(I, J, vals, Nε, Nε)
    end
    env = [sum(B -> sum(abs2, view(nonzeros(B), nzrange(B, i)), init=0.0), Bs) for i in 1:Nε]
    E = Dict(key => [sum(B -> dot(B[:, r[a]], B[:, r[b]]), Bs) for a in eachindex(r), b in eachindex(r)] for (key, r) in rows)
    return env, E
end

"""
    shell_projections(strc, rows, rcut; apply_distortion=false) -> Dict

Projection of the environment of every atom onto the orbitals of its shells: per shell (ion, l) the vector
P[a] = Σ_b T_a(n̂_b) f_cut(r_b) / Σ_b f_cut(r_b) over the neighbours b within `rcut`, with the normalized angular function
T_a = √(4π/(2l+1)) Y_a of the a-th orbital of `shell_types(l)`. An orbital with coefficients u has the projection u·P;
for odd l it vanishes at a centrosymmetric site (the on-site analogue of the projection onto the bond).
"""
function shell_projections(strc, rows, rcut; apply_distortion=false)
    rs_ion = get_ion_positions(strc.ions)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    neighbors = [Tuple{SVector{3, Float64}, Float64}[] for _ in strc.ions]
    for (grid_point, inds) in strc.point_grid.dict0, nn_grid_point in nn_grid_points(grid_point, strc.point_grid.dictR),
        (jion, R) in strc.point_grid.dictR[nn_grid_point], iion in inds
        ri = rs_ion[iion]; rj = apply_pbc(rs_ion[jion], Ts[:, R])
        if apply_distortion
            ri = ri - strc.ions[iion].dist; rj = rj - strc.ions[jion].dist
        end
        d = normdiff(ri, rj)
        (d > 0 && d ≤ rcut) || continue
        push!(neighbors[iion], (normalize(rj - ri), fcut(d, rcut)))
    end
    return Dict(map(collect(keys(rows))) do (iion, l)
        W = sum(last, neighbors[iion], init=0.0)
        P = W > 0 ? [sum(w * √(4π / (2l + 1)) * t(n̂...) for (n̂, w) in neighbors[iion]) / W for t in shell_types(l)] :
                    zeros(2l + 1)
        (iion, l) => P
    end)
end

"""
    symmetry_variants(orbit_i, loc_i, orbit_j, loc_j, same_shell_onsite) -> Vector{Tuple}

Orbital pairs (u, v) and weights over which the kernel of the element (i, j) is averaged: u runs over the orbit of the
shell of orbital i with the weight (S⁻¹u)·c_i (c_i: its coefficients in the shell), v likewise for j; s and hybrid
orbitals are their own orbit (`nothing`, weight 1). On the same site and in the same shell, the tied pairs (u, u) with
the weight (S^(-1/2)u)·c_i (S^(-1/2)u)·c_j are added as well, which keeps the isotropic part of the on-site block
(the bilinear average alone removes a constant). The first pair is the one of the orbitals themselves.
"""
function symmetry_variants(orbit_i, li, orbit_j, lj, same_shell_onsite)
    side(orbit, loc) = isnothing(orbit) ? [(nothing, 1.0)] :
        [(u, w) for (u, w) in ((u, (orbit.Sinv * u)[loc]) for u in orbit.U) if abs(w) > 1e-12]
    A = side(orbit_i, li); B = side(orbit_j, lj)
    pairs = [(u, v, a * b) for (u, a) in A for (v, b) in B]
    if same_shell_onsite
        for u in orbit_i.U
            s = orbit_i.Shalf * u
            w = s[li] * s[lj]
            abs(w) > 1e-12 && push!(pairs, (u, u, w))
        end
    end
    # the pair of the orbitals themselves first (the stored descriptor)
    own(u, loc) = isnothing(u) || (u[loc] ≈ 1 && count(x -> abs(x) > 1e-12, u) == 1)
    k = findfirst(p -> own(p[1], li) && own(p[2], lj), pairs)
    if !isnothing(k)
        pairs[1], pairs[k] = pairs[k], pairs[1]
    end
    return pairs
end

"""
    symmetric_tb_descriptor(h, V, strc, basis, conf, group) -> (descriptors, SymmetryTerms)

Descriptors (Z1, Z2, d, φ, θ1, θ2, e1, e2) of the orbital pairs of all variants of every matrix element under the point
group `group` (see `symmetry_variants` and the file header of symmetry.jl). For an orbital with coefficients u in its
shell (pure orbitals) the angles and the environment are those of that orbital, which makes the descriptor invariant
under a joint rotation of the structure and the orbitals:

- θ: acos of the normalized angular function along the bond, T(u, n̂) = Σ_a u_a T_a(n̂) (for p orbitals the angle
  between axis and bond); on the same site acos of the environment projection u·P (see `shell_projections`); 0 for s;
  the angle between axis and bond for hybrids.
- φ: for two orbitals of the same l, acos(u·v) (for p orbitals the angle between the axes); p and hybrid orbitals: the
  angle between the axes; 0 otherwise.
- e: the squared environment uᵀ E u (see `shell_environment`), times env_scale.
"""
function symmetric_tb_descriptor(h, V, strc::Structure, basis, conf::Config, group;
                                 rcut=get_ml_rcut(conf),
                                 rcut_tol=get_rcut_tol(conf),
                                 apply_distortion=get_ml_apply_distortion(conf),
                                 apply_orthogonality=get_ml_apply_orthogonality(conf),
                                 env_scale=get_ml_env_scale(conf),
                                 apply_distance_distortion=get_ml_apply_distance_distortion(conf),
                                 strc_scale=get_ml_strc_scale(conf))
    Nε = length(basis); Norb_per_ion = size(basis); NR = size(strc.Rs, 2)
    orbits = shell_orbits(basis, group)
    rows = shell_rows(basis)
    env, E = shell_environment(h, strc, basis, rows, rcut)
    P = shell_projections(strc, rows, rcut, apply_distortion=apply_distortion)

    rs_ion = get_ion_positions(strc.ions)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    TRs = [Ts[:, R] for R in 1:NR]
    offset = cumsum(vcat(0, Norb_per_ion[1:end-1]))
    types = [ion.type for ion in strc.ions]

    # Angle and environment of an orbital (pure: coefficients u in its shell; s and hybrids: u = nothing)
    function θ_of(orb, iion, u, n̂)
        l = orb.type.l
        l == 0 && return 0.0
        l < 0 && return isnothing(n̂) ? 0.0 : calc_angle(orb.axis, n̂)
        T = isnothing(n̂) ? dot(u, P[(iion, l)]) :
            sum(u[a] * √(4π / (2l + 1)) * t(n̂[1], n̂[2], n̂[3]) for (a, t) in enumerate(shell_types(l)))
        return acos(clamp(T, -1.0, 1.0))
    end
    function φ_of(orb_i, u, orb_j, v)
        li = orb_i.type.l; lj = orb_j.type.l
        li ≥ 1 && li == lj && return acos(clamp(dot(u, v), -1.0, 1.0))
        axis(orb, w) = orb.type.l < 0 ? orb.axis : orb.type.l == 1 ? SVector{3}(w) : nothing
        ai = axis(orb_i, u); aj = axis(orb_j, v)
        (isnothing(ai) || isnothing(aj)) && return 0.0
        return calc_angle(ai, aj)
    end
    e_of(orb, iion, i, u) = orb.type.l ≥ 1 ? dot(u, E[(iion, orb.type.l)] * u) : env[i]

    cells = collect(strc.point_grid.dict0)
    parts = tmap(chunks(cells, n=4 * Threads.nthreads())) do chunk
        is = [Int64[] for _ in 1:NR]; js = [Int64[] for _ in 1:NR]; vals = [SVector{8, Float64}[] for _ in 1:NR]
        ws = [Float64[] for _ in 1:NR]
        extra = [Tuple{Int64, Int64, SVector{8, Float64}, Float64}[] for _ in 1:NR]
        for (grid_point, inds) in chunk, nn_grid_point in nn_grid_points(grid_point, strc.point_grid.dictR),
            (jion, R) in strc.point_grid.dictR[nn_grid_point], iion in inds
            (Norb_per_ion[iion] == 0 || Norb_per_ion[jion] == 0) && continue
            ri = rs_ion[iion]
            rj = apply_pbc(rs_ion[jion], TRs[R])
            Δr = normdiff(ri, rj)
            Δr_dist = Δr
            if apply_distortion || apply_distance_distortion
                ri_d = ri - strc.ions[iion].dist
                rj_d = rj - strc.ions[jion].dist
                Δr_dist = normdiff(ri_d, rj_d)
                if apply_distortion
                    ri, rj = ri_d, rj_d
                end
            end
            (Δr ≤ rcut && fcut(Δr_dist, rcut + rcut_tol) > 0) || continue
            Δr_in = (apply_distance_distortion || apply_distortion) ? Δr_dist : Δr
            if apply_distortion || apply_distance_distortion
                Δr_in = Δr_in / rcut * strc_scale
            end
            itype = types[iion]; jtype = types[jion]
            Δr_ang = apply_distortion ? Δr_dist : Δr
            Δrij = Δr_ang > 0 ? normalize(rj - ri) : nothing
            Δrji = Δr_ang > 0 ? normalize(ri - rj) : nothing
            for iorb in 1:Norb_per_ion[iion], jorb in 1:Norb_per_ion[jion]
                i = offset[iion] + iorb; j = offset[jion] + jorb
                orb_i = basis.orbitals[iion][iorb]; orb_j = basis.orbitals[jion][jorb]
                l_i = orb_i.type.l; l_j = orb_j.type.l
                decide_orthogonal(Δr, i, j, l_i, l_j; apply_orthogonality=apply_orthogonality) && continue
                orbit_i = l_i ≥ 1 ? orbits[l_i] : nothing; orbit_j = l_j ≥ 1 ? orbits[l_j] : nothing
                loc_i = l_i ≥ 1 ? shell_index(orb_i) : 1; loc_j = l_j ≥ 1 ? shell_index(orb_j) : 1
                same_shell_onsite = iion == jion && Δr == 0 && l_i == l_j && l_i ≥ 1
                for (v, (u_i, u_j, w)) in enumerate(symmetry_variants(orbit_i, loc_i, orbit_j, loc_j, same_shell_onsite))
                    θ1 = θ_of(orb_i, iion, u_i, Δrij); θ2 = θ_of(orb_j, jion, u_j, Δrji)
                    φ = φ_of(orb_i, u_i, orb_j, u_j)
                    e_i = e_of(orb_i, iion, i, u_i); e_j = e_of(orb_j, jion, j, u_j)
                    orbswap = decide_orbswap(itype, jtype, l_i, e_i, l_j, e_j)
                    angleswap = (θ1 > θ2 && itype == jtype) || (orbswap && itype ≠ jtype)
                    Z1, Z2 = orbswap ? (jtype, itype) : (itype, jtype)
                    θ1, θ2 = angleswap ? (θ2, θ1) : (θ1, θ2)
                    if apply_distortion || apply_distance_distortion
                        φ = φ / 2π * strc_scale
                        θ1 = θ1 / 2π * strc_scale
                        θ2 = θ2 / 2π * strc_scale
                    end
                    e1, e2 = orbswap ? (e_j, e_i) : (e_i, e_j)
                    x = SVector{8, Float64}(Z1, Z2, Δr_in, φ, θ1, θ2, e1 * env_scale, e2 * env_scale)
                    if v == 1
                        push!(is[R], i); push!(js[R], j); push!(vals[R], x); push!(ws[R], w)
                    else
                        push!(extra[R], (i, j, x, w))
                    end
                end
            end
        end
        (is, js, vals, ws, extra)
    end
    descriptors = SparseMatrixCSC{SVector{8, Float64}, Int64}[sparse(reduce(vcat, [p[1][R] for p in parts]), reduce(vcat, [p[2][R] for p in parts]),
                                                               reduce(vcat, [p[3][R] for p in parts]), Nε, Nε) for R in 1:NR]
    base_weights = map(1:NR) do R
        W = sparse(reduce(vcat, [p[1][R] for p in parts]), reduce(vcat, [p[2][R] for p in parts]),
                   reduce(vcat, [p[4][R] for p in parts]), Nε, Nε)
        nonzeros(W)
    end
    variant_terms = map(1:NR) do R
        D = descriptors[R]
        map(reduce(vcat, [p[5][R] for p in parts])) do (i, j, x, w)
            k = nzrange(D, j)[searchsortedfirst(view(rowvals(D), nzrange(D, j)), i)]
            (k, x, w)
        end
    end
    return descriptors, SymmetryTerms(base_weights, variant_terms)
end
