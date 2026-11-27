"""
    get_field_kernel(method)

Radial field kernel `-dk/dr` of the real-space kernel `k` of `method` (`ewald`, `wolf` or `zahn`, see `ewald_real!`): a
unit charge at distance `r` creates the field `ke * f(r, α, rcut)` along the connecting vector, and a dipole `μ` the
potential `ke * f(r, α, rcut) * μ ⋅ r̂`. For Ewald, this is the field of the real-space part only; for Wolf, the shift of
the potential does not change the field; for Zahn, the field also vanishes at `rcut`.
"""
function get_field_kernel(method)
    method ∈ ("ewald", "wolf") && return ewald_field_kernel
    method == "zahn" && return zahn_field_kernel
    error("Unknown Ewald method $method for induced dipoles (polarizability): use ewald, wolf or zahn.")
end
ewald_field_kernel(r, α, rcut) = erfc(α * r) / r^2 + 2α / √π * exp(-(α * r)^2) / r
zahn_field_kernel(r, α, rcut) = ewald_field_kernel(r, α, rcut) - ewald_field_kernel(rcut, α, rcut)

"""
    induced_dipole_potentials(pos, q, lattice, groups, ngroups; rcut, alpha, method, mesh_spacing=0.5, long_range_scale=1.0)

Potentials (V) of the dipoles that the field of the charges `q` induces on the atoms, to first order and per group of
atoms: atom `j` belongs to group `groups[j]` (0: not polarizable) and gets the dipole `μ_j = E_j / ke` of polarizability
1 Å³ in the field `E_j` of all charges (without the induced dipoles). Returns the matrix `Φ` (atoms × `ngroups`) with the
potential `Φ[i, g]` of the dipoles of group `g` at atom `i`; with the polarizabilities `p_g` (Å³) the potential of all
induced dipoles is `Φ * p`. Field and dipole potential use the damped real-space kernel of `method` within `rcut`, on a
neighbor grid (linear in the number of atoms). For `method = ewald`, the reciprocal-space parts of the field
(`ewald_recip_field`) and of the dipole potential (`ewald_recip_dipole_potentials`, mesh `mesh_spacing`) are added, both
multiplied by `long_range_scale` (as the potential of the charges in `ewald_sum`): with `long_range_scale = 1` this is
the full Ewald sum of field and dipole potential (tin-foil boundary conditions), with 0 the real-space part only.
"""
function induced_dipole_potentials(pos, q, lattice, groups, ngroups; rcut, alpha, method, mesh_spacing=0.5, long_range_scale=1.0)
    method = lowercase(method)
    f = get_field_kernel(method)
    long_range = method == "ewald" && long_range_scale ≠ 0
    N = length(pos)
    Rs = ewald_translation_vectors(pos, lattice, rcut)
    point_grid = PointGrid(reduce(hcat, pos), frac_to_cart(Rs, lattice), grid_size=rcut)
    Ts = [SVector{3, Float64}(T) for T in eachcol(frac_to_cart(Rs, lattice))]
    R0 = [iszero(R) for R in eachcol(Rs)]
    rs = [SVector{3, Float64}(r) for r in pos]
    E = zeros(SVector{3, Float64}, N)
    Φ = zeros(Float64, N, ngroups)
    cells = collect(point_grid.dict0)
    function pairs(action, cell)
        grid_point, inds = cell
        for nn_grid_point in nn_grid_points(grid_point, point_grid.dictR), (j, R) in point_grid.dictR[nn_grid_point], i in inds
            (i == j && R0[R]) && continue
            d = rs[i] - apply_pbc(rs[j], Ts[R])
            r = norm(d)
            (r < rcut && r > 1e-14) && action(i, j, d, r)
        end
    end
    tforeach(cells) do cell
        pairs(cell) do i, j, d, r
            groups[i] > 0 && (E[i] += ke * q[j] * f(r, alpha, rcut) / r * d)
        end
    end
    if long_range
        lattice_inv = inv(lattice)
        green, Gc = pme_green(lattice_inv, alpha, mesh_from_spacing(lattice, mesh_spacing))
        E .+= ke * long_range_scale .* ewald_recip_field(pos, q, lattice, lattice_inv, green, Gc) .* (groups .> 0)
    end
    tforeach(cells) do cell
        pairs(cell) do i, j, d, r
            groups[j] > 0 && (Φ[i, groups[j]] += f(r, alpha, rcut) / r * dot(E[j], d))
        end
    end
    if long_range
        for g in 1:ngroups
            μ = [groups[j] == g ? E[j] : zero(E[j]) for j in 1:N]
            Φ[:, g] .+= long_range_scale .* ewald_recip_dipole_potentials(pos, μ, lattice, lattice_inv, green, Gc)
        end
    end
    return Φ
end

"""
    DipolePair(j, d, B, C, fr)

One real-space neighbour of an atom (an entry of `dipole_pair_list`): atom index `j` (with its periodic image),
displacement `d = r_i - r_j`, and the precomputed radial factors of the Ewald dipole field,
`B(r) = [erfc(αr) + (2αr/√π) e^{-α²r²}] / r³` and `C(r) = [3 erfc(αr) + (2αr/√π)(3 + 2α²r²) e^{-α²r²}] / r⁵`
(field of the dipole `μ_j`: `ke [C (μ_j⋅d) d - B μ_j]`), and of the charge field and dipole potential
`fr = f(r, α, rcut) / r` (field of a unit charge `ke fr d`, potential of `μ_j`: `fr (μ_j⋅d)`).
"""
struct DipolePair
    j :: Int32
    d :: SVector{3, Float64}
    B :: Float64
    C :: Float64
    fr :: Float64
end

"""
    dipole_pair_list(pos, lattice, Rs, point_grid, rcut, alpha)

Real-space neighbour pairs of all atoms within `rcut` (periodic images `Rs` on the neighbour grid `point_grid`), one
list of `DipolePair` per atom, with the radial factors of the Ewald field and dipole potential precomputed. The pair of
an atom with itself in the home cell is excluded; its periodic images are included. Linear in the number of atoms.
"""
function dipole_pair_list(pos, lattice, Rs, point_grid, rcut, alpha)
    rs = [SVector{3, Float64}(r) for r in pos]
    Ts = [SVector{3, Float64}(T) for T in eachcol(frac_to_cart(Rs, lattice))]
    R0 = [iszero(R) for R in eachcol(Rs)]
    pairs = [DipolePair[] for _ in eachindex(pos)]
    for (grid_point, inds) in point_grid.dict0, nn_grid_point in nn_grid_points(grid_point, point_grid.dictR),
            (j, R) in point_grid.dictR[nn_grid_point], i in inds
        (i == j && R0[R]) && continue
        d = rs[i] - apply_pbc(rs[j], Ts[R])
        r = norm(d)
        (r < rcut && r > 1e-14) || continue
        x = alpha * r
        ex = exp(-x^2)
        B = (erfc(x) + 2x / √π * ex) / r^3
        C = (3 * erfc(x) + 2x / √π * (3 + 2x^2) * ex) / r^5
        push!(pairs[i], DipolePair(j, d, B, C, ewald_field_kernel(r, alpha, rcut) / r))
    end
    return pairs
end

# real-space field (V/Å) of the dipoles μ (e Å) at all atoms: E_i = ke Σ_j [C (μ_j⋅d) d - B μ_j]
function _dipole_field_real!(E, pairs, μ)
    tforeach(eachindex(pairs)) do i
        Ei = zero(SVector{3, Float64})
        for pr in pairs[i]
            Ei += ke * (pr.C * dot(μ[pr.j], pr.d) * pr.d - pr.B * μ[pr.j])
        end
        E[i] = Ei
    end
    return E
end

# add the reciprocal-space field of the dipoles and the self term (both × long_range_scale) to E
function _add_recip_dipole_field!(E, pos, μ, lattice, lattice_inv, green, Gc, alpha, long_range_scale)
    long_range_scale == 0 && return E
    E .+= ke * long_range_scale .* ewald_recip_dipole_field(pos, μ, lattice, lattice_inv, green, Gc)
    # the reciprocal sum contains the field -ke 4α³/(3√π) μ_i of each Gaussian dipole at its own center: subtract it
    E .+= ke * long_range_scale * (4 * alpha^3 / (3√π)) .* μ
    return E
end

"""
    ewald_dipole_field(pos, μ, lattice; rcut, alpha, mesh_spacing=0.5, long_range_scale=1.0, Rs=nothing, point_grid=nothing, pairs=nothing)

Electric field (V/Å) at all atoms of the point dipoles `μ` (e Å, one 3-vector per atom) over all periodic images, full
Ewald sum with tin-foil boundary conditions, without the interaction of an atom with its own dipole in the home cell:
the real-space part on the neighbour grid (`pairs` from `dipole_pair_list`, built if not given),
`E_i += ke Σ_j [C(r) (μ_j⋅d) d - B(r) μ_j]` with `d = r_i - r_j` and the `B`, `C` of `DipolePair`, plus the
reciprocal-space part (`ewald_recip_dipole_field` on the PME mesh `mesh_spacing`) and the self term: the reciprocal sum
contains the field `-ke (4α³ / 3√π) μ_i` of each Gaussian dipole at its own center, which is subtracted here. Both
reciprocal-space parts and the self term are multiplied by `long_range_scale` (as in `ewald_sum`). Linear in the
number of atoms.
"""
function ewald_dipole_field(pos, μ, lattice; rcut, alpha, mesh_spacing=0.5, long_range_scale=1.0,
                            Rs=nothing, point_grid=nothing, pairs=nothing)
    N = length(pos)
    if isnothing(Rs); Rs = ewald_translation_vectors(pos, lattice, rcut); end
    if isnothing(point_grid); point_grid = PointGrid(reduce(hcat, pos), frac_to_cart(Rs, lattice), grid_size=rcut); end
    if isnothing(pairs); pairs = dipole_pair_list(pos, lattice, Rs, point_grid, rcut, alpha); end
    E = _dipole_field_real!(zeros(SVector{3, Float64}, N), pairs, μ)
    if long_range_scale ≠ 0
        lattice_inv = inv(lattice)
        green, Gc = pme_green(lattice_inv, alpha, mesh_from_spacing(lattice, mesh_spacing))
        _add_recip_dipole_field!(E, pos, μ, lattice, lattice_inv, green, Gc, alpha, long_range_scale)
    end
    return E
end

"""
    self_consistent_dipoles(pos, q, lattice, groups, p; rcut, alpha, mesh_spacing=0.5, long_range_scale=1.0,
                            charge_long_range_scale=long_range_scale, dipole_tol=1e-8, dipole_maxiter=100,
                            Rs=nothing, point_grid=nothing, verbosity=0)

Self-consistent induced dipoles (full Ewald sum, tin-foil boundary) of the polarizable atoms: atom `i` with
`groups[i] > 0` has the polarizability `p[groups[i]]` (Å³) and gets the dipole

    μ_i = p_i / ke ( E⁰_i + Σ_j T_ij μ_j )

in the field `E⁰` of the charges `q` (V/Å) and of all other dipoles (`ewald_dipole_field`). The symmetric system
`(ke/p_i) μ_i - Σ_j T_ij μ_j = E⁰_i` over the polarizable atoms is solved with preconditioned conjugate gradient
(Jacobi preconditioner `p_i / ke`, matrix-free products from the cached pair list `dipole_pair_list`), starting from
the first-order solution, until the relative residual `‖r‖ / ‖E⁰‖` is below `dipole_tol` (error after
`dipole_maxiter` iterations, usually a polarization catastrophe). Atoms with `groups[i] == 0` carry no dipole but feel
the field of the dipoles. The reciprocal-space parts of the dipole tensor and of the returned dipole potential `Φ`
(V, at all atoms) are multiplied by `long_range_scale`, those of the charge field `E⁰` by
`charge_long_range_scale` (default: the same). Returns the dipoles `μ` (e Å, zero on non-polarizable atoms), `Φ`,
and the number of iterations.
"""
function self_consistent_dipoles(pos, q, lattice, groups, p; rcut, alpha, mesh_spacing=0.5, long_range_scale=1.0,
                                 charge_long_range_scale=long_range_scale, dipole_tol=1e-8, dipole_maxiter=100,
                                 Rs=nothing, point_grid=nothing, verbosity=0)
    N = length(pos)
    μ = zeros(SVector{3, Float64}, N)
    pol = [i for i in 1:N if groups[i] > 0 && p[groups[i]] != 0]   # zero polarizability: no dipole
    isempty(pol) && return μ, zeros(N), 0
    if isnothing(Rs); Rs = ewald_translation_vectors(pos, lattice, rcut); end
    if isnothing(point_grid); point_grid = PointGrid(reduce(hcat, pos), frac_to_cart(Rs, lattice), grid_size=rcut); end
    pairs = dipole_pair_list(pos, lattice, Rs, point_grid, rcut, alpha)
    lattice_inv = inv(lattice)
    green, Gc = long_range_scale == 0 && charge_long_range_scale == 0 ? (nothing, nothing) :
        pme_green(lattice_inv, alpha, mesh_from_spacing(lattice, mesh_spacing))

    # field E⁰ of the charges at all atoms (V/Å)
    E0 = zeros(SVector{3, Float64}, N)
    tforeach(eachindex(pos)) do i
        Ei = zero(SVector{3, Float64})
        for pr in pairs[i]
            Ei += ke * q[pr.j] * pr.fr * pr.d
        end
        E0[i] = Ei
    end
    charge_long_range_scale ≠ 0 && (E0 .+= ke * charge_long_range_scale .* ewald_recip_field(pos, q, lattice, lattice_inv, green, Gc))

    # matrix-free product of A = ke/p - T on the polarizable atoms: (A x)_i = ke/p_i x_i - E_dip,i(x)
    n = length(pol)
    pvec = [p[groups[i]] for i in pol]
    b = [E0[i] for i in pol]
    function Amul!(y, x)
        for (k, i) in enumerate(pol); μ[i] = x[k]; end
        E = _dipole_field_real!(zeros(SVector{3, Float64}, N), pairs, μ)
        _add_recip_dipole_field!(E, pos, μ, lattice, lattice_inv, green, Gc, alpha, long_range_scale)
        for (k, i) in enumerate(pol); y[k] = ke / pvec[k] * x[k] - E[i]; end
        return y
    end

    vdot(a, b) = sum(dot.(a, b))
    x = [pvec[k] / ke * b[k] for k in 1:n]                  # first-order start
    r = b .- Amul!(similar(b), x)
    z = [r[k] * (pvec[k] / ke) for k in 1:n]                  # Jacobi preconditioner
    d = copy(z)
    rz = vdot(r, z)
    normb = sqrt(vdot(b, b))
    it = 0
    while sqrt(vdot(r, r)) > dipole_tol * normb
        it ≥ dipole_maxiter && error("self_consistent_dipoles did not converge in $dipole_maxiter iterations (relative " *
            "residual $(sqrt(vdot(r, r)) / normb) > dipole_tol = $dipole_tol): polarization catastrophe, use smaller polarizabilities.")
        Ap = Amul!(similar(d), d)
        γ = rz / vdot(d, Ap)
        x .+= γ .* d
        r .-= γ .* Ap
        z = [r[k] * (pvec[k] / ke) for k in 1:n]
        rz_new = vdot(r, z)
        d .= z .+ (rz_new / rz) .* d
        rz = rz_new
        it += 1
    end
    verbosity > 1 && println("self-consistent induced dipoles: $it CG iterations, relative residual $(sqrt(vdot(r, r)) / normb)")
    for (k, i) in enumerate(pol); μ[i] = x[k]; end

    # potential of the converged dipoles at all atoms (V)
    Φ = zeros(N)
    tforeach(eachindex(pos)) do i
        acc = 0.0
        for pr in pairs[i]
            acc += pr.fr * dot(μ[pr.j], pr.d)
        end
        Φ[i] = ke * acc
    end
    long_range_scale ≠ 0 && (Φ .+= ke * long_range_scale .* ewald_recip_dipole_potentials(pos, μ, lattice, lattice_inv, green, Gc))
    return μ, Φ, it
end

"""
    ewald_recip_dipole_field(pos, μ, box, box_inv, green, Gc)

Reciprocal-space part of the electric field (Gaussian units, e/Å²) of the point dipoles `μ` (e Å, one 3-vector per
atom) at the atoms, with the factors `green` and `Gc` of `pme_green`: the dipole components are spread on the grid, the
charge density of the dipoles is `ρ(G) = -i G ⋅ P(G)` (see `ewald_recip_dipole_potentials`), and `-i G φ(G)` of its
potential is transformed back per Cartesian component and interpolated to the atoms (as in `ewald_recip_field`).
Includes the field of each Gaussian dipole at its own center; subtract it with the self term of `ewald_dipole_field`.
"""
function ewald_recip_dipole_field(pos, μ, box, box_inv, green, Gc)
    rho_k = _dipole_density_k!(zeros(ComplexF64, size(green)), pos, μ, box, box_inv, Gc)
    return _recip_field_from_potential(green .* rho_k, pos, box_inv, Gc)
end

# charge density ρ(G) = -i G ⋅ P(G) of the dipoles μ spread on the PME grid
function _dipole_density_k!(rho_k, pos, μ, box, box_inv, Gc)
    P = zeros(Float64, size(rho_k))
    for c in 1:3
        _spread_bspline4!(P, pos, [m[c] for m in μ], box, box_inv)
        rho_k .-= im .* Gc[c] .* fft(P)
    end
    return rho_k
end

"""
    ewald_recip_dipole_potentials(pos, μ, box, box_inv, green, Gc)

Reciprocal-space part of the potential (Gaussian units, e/Å) of the point dipoles `μ` (e Å, one 3-vector per atom) at
the atoms, with the factors `green` and `Gc` of `pme_green`: the dipole components are spread on the grid, the charge
density of the dipoles is `ρ(G) = -i G ⋅ P(G)`, and its potential is interpolated to the atoms. Without self term (the
potential of a Gaussian dipole vanishes at its center).
"""
function ewald_recip_dipole_potentials(pos, μ, box, box_inv, green, Gc)
    rho_k = _dipole_density_k!(zeros(ComplexF64, size(green)), pos, μ, box, box_inv, Gc)
    return _gather_bspline4(real(ifft(green .* rho_k)), pos, box_inv)
end
