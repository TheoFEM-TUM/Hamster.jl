# --------------------------------
# Real-space short-range Ewald
# --------------------------------
function _ewald_real!(
    phi,
    pos,
    q,
    box,
    box_inv,
    alpha,
    rcut
)
    N = length(pos)
    fill!(phi, 0.0)
    E = 0.0

    @views for i in 1:N-1
        ri = pos[i]
        qi = q[i]

        for j in i+1:N
            rj = pos[j]
            qj = q[j]

            dr = _disp_pbc_general(ri, rj, box, box_inv)
            r = norm(dr)

            if r < rcut && r > 1e-14
                v = erfc(alpha * r) / r
                phi[i] += qj * v
                phi[j] += qi * v
                E += qi * qj * v
            end
        end
    end

    return E
end

# --------------------------------
# Reciprocal-space SPME-like part
# --------------------------------
function _recip!(
    phi,
    pos,
    q,
    box,
    box_inv,
    alpha,
    mesh_spacing
)

    mesh = mesh_from_spacing(box, mesh_spacing)

    Nx, Ny, Nz = mesh
    V = abs(det(box))

    rho = zeros(Float64, Nx, Ny, Nz)
    _spread_bspline4!(rho, pos, q, box, box_inv)

    rho_k = fft(rho)
    phi_k = similar(rho_k)

    # reciprocal lattice matrix: columns are reciprocal lattice vectors
    Gmat = 2π * transpose(box_inv)

    # approximate grid spacings along reciprocal directions for spline deconvolution
    # uses the norms of the direct lattice vectors divided by grid counts
    a1 = box[:, 1]
    a2 = box[:, 2]
    a3 = box[:, 3]
    h1 = norm(a1) / Nx
    h2 = norm(a2) / Ny
    h3 = norm(a3) / Nz

    p = 4  # cubic B-spline order

    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        ni = (i - 1 <= Nx ÷ 2) ? (i - 1) : (i - 1 - Nx)
        nj = (j - 1 <= Ny ÷ 2) ? (j - 1) : (j - 1 - Ny)
        nk = (k - 1 <= Nz ÷ 2) ? (k - 1) : (k - 1 - Nz)

        nvec = SVector{3}(ni, nj, nk)
        G = Gmat * nvec
        G2 = dot(G, G)

        if G2 < 1e-30
            phi_k[i, j, k] = 0.0 + 0.0im
            continue
        end

        # simple spline deconvolution factor
        # for a general cell this is approximate but usually good enough for a research implementation
        bx = _sinc(0.5 * (2π * ni / Nx))^p
        by = _sinc(0.5 * (2π * nj / Ny))^p
        bz = _sinc(0.5 * (2π * nk / Nz))^p
        b2 = (bx * by * bz)^2

        if b2 < 1e-28
            phi_k[i, j, k] = 0.0 + 0.0im
            continue
        end

        green = 4π * exp(-G2 / (4 * alpha^2)) / G2
        phi_k[i, j, k] = green * rho_k[i, j, k] / b2
    end

    phi_grid = real(ifft(phi_k))
    phi .= _gather_bspline4(phi_grid, pos, box_inv)

    voxelvol = V / (Nx * Ny * Nz)
    E = 0.5 * sum(rho .* phi_grid) * voxelvol

    return E
end

# --------------------------------
# Main function
# --------------------------------
function pme_bspline(pos, q, box;
    rcut::Float64,
    alpha=nothing,
    alpha_factor=4.0,
    mesh_spacing=0.5,
    include_short::Bool=true,
    include_long::Bool=true,
    subtract_self::Bool=true,
)

    @assert length(q) == length(pos) "length(q) must equal number of atoms"
    @assert abs(sum(q)) < 1e-8 "Periodic PME requires a neutral cell unless you add a neutralizing background"

    box_inv = inv(box)
    α = isnothing(alpha) ? (alpha_factor / rcut) : alpha

    phi_s = zeros(Float64, length(pos))
    phi_l = zeros(Float64, length(pos))

    Es = include_short ? ke * _ewald_real!(phi_s, pos, q, box, box_inv, α, rcut)    : 0.0
    El = include_long  ? ke * _recip!(phi_l, pos, q, box, box_inv, α, mesh_spacing) : 0.0

    phi = ke .* (phi_s .+ phi_l)

    Eself = 0.0
    if subtract_self
        c = ke * α / sqrt(pi)
        Eself = -c * sum(q .^ 2)
        phi .-= 2c .* q
    end

    return (
        alpha = α,
        potentials = phi,
        potentials_short = phi_s,
        potentials_long = phi_l,
        energy = Es + El + Eself,
        energy_short = Es,
        energy_long = El,
        energy_self = Eself,
    )
end