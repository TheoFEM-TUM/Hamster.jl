"""
    ewald_sum(pos, q, lattice; kwargs...)

Compute electrostatic potentials and energies using Ewald-type methods
(Ewald, Wolf, Zahn, etc.).

This function combines real-space and reciprocal-space contributions
(depending on `method`) and returns per-site potentials together with
energy components.

# Arguments
- `pos`: Atomic positions
- `q`: Charges (must sum to ~0 for periodic systems)
- `lattice`: 3×3 lattice matrix

# Keyword Arguments
- `rcut`: Real-space cutoff
- `Rs`: Translation vectors (optional, generated if not provided)
- `point_grid`: Neighbor grid (optional, generated if not provided)
- `alpha`: Ewald screening parameter (overrides `alpha_factor` if nonzero)
- `alpha_factor`: Sets `alpha ≈ alpha_factor / rcut` if `alpha == 0`
- `mesh_spacing`: Grid spacing for reciprocal-space PME
- `method`: `"ewald"`, `"wolf"`, `"zahn"`, etc.
- `include_short`: Include real-space contribution
- `include_long`: Include reciprocal-space contribution (disabled for Wolf/Zahn by default)
- `subtract_self`: Subtract Ewald self-interaction (disabled for Wolf/Zahn)
- `long_range_scale`: Factor of the reciprocal-space contribution and the self-interaction correction, i.e. of the
  potential of the Gaussian charge clouds of width 1/α (e.g. 1/ε∞ for a long range screened by the electrons)

# Returns
Named tuple with:
- `potentials`: Total site potentials
- `potentials_short`, `potentials_long`: Real and reciprocal contributions
- `energy`: Total electrostatic energy
- `energy_short`, `energy_long`, `energy_self`: Individual energy components
- `alpha`: Effective screening parameter used
- `grid_time`, `time_short`, `time_long`: Timing diagnostics

# Methods
- `"ewald"`:
    Full Ewald summation (real + reciprocal + self correction)

- `"wolf"` / `"zahn"`:
    Truncated real-space approximations using screened Coulomb kernels.
    No reciprocal-space term or self correction is applied.
"""
function ewald_sum(pos, q, lattice;
    rcut,
    Rs=nothing,
    point_grid=nothing,
    alpha=0.,
    alpha_factor=4.0,
    mesh_spacing=0.5,
    method="ewald",
    include_short::Bool=true,
    include_long::Bool=!(method in ("zahn", "wolf")),
    subtract_self::Bool=!(method in ("zahn", "wolf")),
    long_range_scale=1.0
)

    @assert length(q) == length(pos) "length(q) must equal number of atoms"
    lowercase(method) ∈ ("wolf", "zahn") || abs(sum(q)) < 1e-8 ||
        error("The Ewald sum (method = ewald) needs a neutral cell: the charges add up to $(sum(q)). Use method = wolf or zahn.")

    lattice_inv = inv(lattice)
    α = alpha == 0. ? (alpha_factor / rcut) : alpha

    grid_time = @elapsed begin
        if isnothing(Rs)
            Rs = ewald_translation_vectors(pos, lattice, rcut)
        end
        if isnothing(point_grid)
            point_grid = PointGrid(reduce(hcat, pos), frac_to_cart(Rs, lattice), grid_size=rcut)
        end
    end

    phi_s = zeros(Float64, length(pos))
    phi_l = zeros(Float64, length(pos))

    sr_time = @elapsed Es = include_short ? ke * ewald_real!(phi_s, pos, q, Rs, lattice, point_grid, α, rcut, method=method)  : 0.0
    lr_time = @elapsed El = include_long  ? ke * ewald_recip!(phi_l, pos, q, lattice, lattice_inv, α, mesh_spacing)                     : 0.0

    phi = ke .* (phi_s .+ long_range_scale .* phi_l)

    Eself = 0.0
    if subtract_self
        c = long_range_scale * ke * α / sqrt(pi)
        Eself = -c * sum(q .^ 2)
        phi .-= 2c .* q
    end

    return (
        alpha = α,
        potentials = phi,
        potentials_short = phi_s,
        potentials_long = phi_l,
        energy = Es + long_range_scale * El + Eself,
        energy_short = Es,
        energy_long = long_range_scale * El,
        energy_self = Eself,
        grid_time = grid_time,
        time_short = sr_time,
        time_long = lr_time
    )
end

"""
    ewald_translation_vectors(pos, lattice, rcut)

Lattice translations (fractional) of all periodic images that can contain atoms within `rcut` of an atom of the
cell. Instead of checking all atom pairs (as `get_translation_vectors`, which is very slow for large cells and large
cutoffs), images are discarded with a bounding-sphere estimate: the atoms lie within a sphere of radius ρ around their
center, so an image translated by T can only contain neighbors if |T| ≤ rcut + 2ρ. The result can contain a few
translations without neighbors, which do not change the sum.
"""
function ewald_translation_vectors(pos, lattice, rcut)
    Rs = get_translation_vectors(get_Rmax(lattice, rcut=rcut, positions=reduce(hcat, pos)))
    center = sum(pos) / length(pos)
    ρ = maximum(r -> norm(r - center), pos)
    Ts = frac_to_cart(Rs, lattice)
    return Rs[:, [norm(Ts[:, R]) ≤ rcut + 2ρ for R in axes(Rs, 2)]]
end

"""
    ewald_real!(phi, pos, qs, Rs, lattice, point_grid, alpha, rcut; method="ewald")

Compute the real-space electrostatic potential and energy in-place using
an Ewald-type kernel.

The potential `phi` is accumulated for each atom from pairwise interactions
within a cutoff `rcut`, including periodic images defined by `Rs` and `lattice`.

# Arguments
- `phi`: Output vector of site potentials (modified in-place)
- `pos`: Atomic positions
- `qs`: Charges
- `Rs`: Lattice translation vectors (fractional coordinates)
- `lattice`: Lattice matrix
- `point_grid`: Neighbor iteration structure
- `alpha`: Screening parameter
- `rcut`: Real-space cutoff radius
- `method`: Choice of real-space kernel (`"ewald"`, `"wolf"`, `"zahn"`, ...)

# Returns
- Total electrostatic energy `E = 0.5 * (qs ⋅ phi)`

# Kernels
The interaction kernel is selected via `method`:

- `"ewald"`:
    Uses the standard real-space Ewald term

        erfc(α r) / r

    This is only one part of the full Ewald sum and must be combined with
    reciprocal-space and self-interaction corrections for exact results.

- `"wolf"`:
    Truncated and shifted Ewald kernel

        erfc(α r) / r - erfc(α rcut) / rcut

    Ensures the potential goes to zero at `rcut`. Approximates long-range
    electrostatics without reciprocal-space contributions.

- `"zahn"`:
    Damped shifted force (Zahn, Fennell and Gezelter)

        erfc(α r) / r - erfc(α rcut) / rcut + (erfc(α rcut) / rcut² + 2α/√π exp(-α² rcut²) / rcut) (r - rcut)

    Potential and force go to zero at `rcut`.
"""
function ewald_real!(phi, pos, qs, Rs, lattice, point_grid, alpha, rcut; method="ewald")

    kernel = get_realspace_kernel(method)
    fill!(phi, 0.0)

    Ts = [SVector{3, Float64}(T) for T in eachcol(frac_to_cart(Rs, lattice))]
    R0 = [iszero(R) for R in eachcol(Rs)]

    # Pairs are visited grid cell by grid cell (neighbor cells and periodic images are unique, so every pair is
    # visited once). Every atom belongs to exactly one cell, so the cells can be processed in parallel.
    tforeach(collect(point_grid.dict0)) do (grid_point, inds)
        for nn_grid_point in nn_grid_points(grid_point, point_grid.dictR), (j, R) in point_grid.dictR[nn_grid_point], i in inds
            (i == j && R0[R]) && continue
            r = normdiff(pos[i], pos[j], Ts[R])
            if r < rcut && r > 1e-14
                phi[i] += qs[j] * kernel(r, alpha, rcut)
            end
        end
    end

    E = 0.5 * (qs ⋅ phi)

    return E
end

function get_realspace_kernel(method)
    if method == "ewald"
        return ewald_kernel
    elseif method == "wolf"
        return wolf_kernel
    elseif method == "zahn"
        return zahn_kernel
    else
        error("Unknown real-space method: $method. Use \"ewald\", \"wolf\", or \"zahn\".")
    end
end
ewald_kernel(r, α, rcut) = erfc(α * r) / r
wolf_kernel(r, α, rcut) = ewald_kernel(r, α, rcut) - erfc(α * rcut) / rcut
# damped shifted force (Fennell and Gezelter, J. Chem. Phys. 124, 234104 (2006)): value and derivative vanish at rcut
zahn_kernel(r, α, rcut) = ewald_kernel(r, α, rcut) - erfc(α * rcut) / rcut +
                          (erfc(α * rcut) / rcut^2 + 2α / √π * exp(-(α * rcut)^2) / rcut) * (r - rcut)

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
    induced_dipole_potentials(pos, q, lattice, groups, ngroups; rcut, alpha, method)

Potentials (V) of the dipoles that the field of the charges `q` induces on the atoms, to first order and per group of
atoms: atom `j` belongs to group `groups[j]` (0: not polarizable) and gets the dipole `μ_j = E_j / ke` of polarizability
1 Å³ in the field `E_j` of all charges (without the induced dipoles). Returns the matrix `Φ` (atoms × `ngroups`) with the
potential `Φ[i, g]` of the dipoles of group `g` at atom `i`; with the polarizabilities `p_g` (Å³) the potential of all
induced dipoles is `Φ * p`. Field and dipole potential use the damped real-space kernel of `method` (`wolf` or `zahn`)
within `rcut`, on a neighbor grid (linear in the number of atoms).
"""
function induced_dipole_potentials(pos, q, lattice, groups, ngroups; rcut, alpha, method)
    f = get_field_kernel(lowercase(method))
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
    tforeach(cells) do cell
        pairs(cell) do i, j, d, r
            groups[j] > 0 && (Φ[i, groups[j]] += f(r, alpha, rcut) / r * dot(E[j], d))
        end
    end
    return Φ
end

"""
    ewald_recip!(phi, pos, q, box, box_inv, alpha, mesh_spacing)

Compute the reciprocal-space contribution to the electrostatic potential
and energy using a Particle Mesh Ewald (PME) scheme with cubic B-splines.

The potential `phi` is updated in-place for each atomic position by solving
Poisson’s equation on a regular grid in reciprocal space.

# Arguments
- `phi`: Output vector of site potentials (modified in-place)
- `pos`: Atomic positions
- `q`: Charges
- `box`: 3×3 lattice matrix
- `box_inv`: Inverse of the lattice matrix
- `alpha`: Ewald screening parameter
- `mesh_spacing`: Target real-space grid spacing

# Returns
- Reciprocal-space electrostatic energy

# Method
1. Charges are interpolated onto a 3D grid using 4th-order B-splines.
2. The charge density is Fourier transformed (`fft`).
3. The Poisson equation is solved in reciprocal space using the Ewald Green’s function:

       G(k) = 4π exp(-k² / (4α²)) / k²

4. A spline deconvolution factor is applied to correct for interpolation smoothing.
5. The potential is transformed back to real space (`ifft`) and interpolated to atomic positions.
6. The reciprocal-space energy is computed from the grid representation.
"""
function ewald_recip!(phi, pos, q, box, box_inv, alpha, mesh_spacing)

    mesh = mesh_from_spacing(box, mesh_spacing)

    Nx, Ny, Nz = mesh
    V = abs(det(box))

    rho = zeros(Float64, Nx, Ny, Nz)
    _spread_bspline4!(rho, pos, q, box, box_inv)

    rho_k = fft(rho)
    phi_k = similar(rho_k)

    # reciprocal lattice vectors (columns of 2π box⁻ᵀ)
    Gmat = SMatrix{3, 3}(2π * transpose(box_inv))
    g1, g2, g3 = Gmat[:, 1], Gmat[:, 2], Gmat[:, 3]

    p = 4  # cubic B-spline order
    freq(i, N) = (i - 1 <= N ÷ 2) ? (i - 1) : (i - 1 - N)
    # spline deconvolution factors, separable in the three directions
    bx = [_sinc(π * freq(i, Nx) / Nx)^p for i in 1:Nx]
    by = [_sinc(π * freq(j, Ny) / Ny)^p for j in 1:Ny]
    bz = [_sinc(π * freq(k, Nz) / Nz)^p for k in 1:Nz]
    inv_4α² = 1 / (4 * alpha^2)

    # Column-major loop order, threaded over the planes k (each task writes its own planes)
    tforeach(1:Nz) do k
        Gk = freq(k, Nz) * g3
        for j in 1:Ny
            Gjk = Gk + freq(j, Ny) * g2
            for i in 1:Nx
                G = Gjk + freq(i, Nx) * g1
                G2 = dot(G, G)
                b2 = (bx[i] * by[j] * bz[k])^2
                if G2 < 1e-30 || b2 < 1e-28
                    phi_k[i, j, k] = 0.0 + 0.0im
                else
                    phi_k[i, j, k] = 4π * exp(-G2 * inv_4α²) / G2 * rho_k[i, j, k] / b2
                end
            end
        end
    end

    phi_grid = real(ifft(phi_k))
    phi .= _gather_bspline4(phi_grid, pos, box_inv)

    voxelvol = V / (Nx * Ny * Nz)
    E = 0.5 * sum(rho .* phi_grid) * voxelvol

    return E
end