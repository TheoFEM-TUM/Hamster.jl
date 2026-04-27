using FFTW, SpecialFunctions
using LinearAlgebra

const ke = 14.3996454784255

@inline _wrap01(x) = x - floor(x)
@inline _sinc(x) = abs(x) < 1e-14 ? 1.0 : sin(x)/x

# cubic B-spline weights
# returns weights for offsets (-1, 0, 1, 2)
@inline function _bspline4_weights(t)
    w0 = (1 - t)^3 / 6
    w1 = (3t^3 - 6t^2 + 4) / 6
    w2 = (-3t^3 + 3t^2 + 3t + 1) / 6
    w3 = t^3 / 6
    return (w0, w1, w2, w3)
end

# minimum-image displacement for a general 3x3 lattice box
@inline function _disp_pbc_general(ri::AbstractVector, rj::AbstractVector, box::AbstractMatrix, box_inv::AbstractMatrix)
    dr = ri - rj
    s = box_inv * dr
    s .-= round.(s)
    return box * s
end

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
    N = size(pos, 2)
    fill!(phi, 0.0)
    E = 0.0

    for i in 1:N-1
        ri = @view pos[:, i]
        qi = q[i]

        for j in i+1:N
            rj = @view pos[:, j]
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
# Charge spreading with cubic B-splines
# --------------------------------
function _spread_bspline4!(
    rho,
    pos,
    q,
    box,
    box_inv
)
    Nx, Ny, Nz = size(rho)
    fill!(rho, 0.0)

    cellvol = abs(det(box))
    voxelvol = cellvol / (Nx * Ny * Nz)

    N = size(pos, 2)

    for i in 1:N
        r = @view pos[:, i]
        qi = q[i]

        s = box_inv * r
        sx = _wrap01(s[1]) * Nx
        sy = _wrap01(s[2]) * Ny
        sz = _wrap01(s[3]) * Nz

        ix = floor(Int, sx)
        iy = floor(Int, sy)
        iz = floor(Int, sz)

        tx = sx - ix
        ty = sy - iy
        tz = sz - iz

        wx = _bspline4_weights(tx)
        wy = _bspline4_weights(ty)
        wz = _bspline4_weights(tz)

        for (ax, ox) in enumerate((-1, 0, 1, 2))
            gx = mod(ix + ox, Nx) + 1
            for (ay, oy) in enumerate((-1, 0, 1, 2))
                gy = mod(iy + oy, Ny) + 1
                for (az, oz) in enumerate((-1, 0, 1, 2))
                    gz = mod(iz + oz, Nz) + 1
                    rho[gx, gy, gz] += qi * wx[ax] * wy[ay] * wz[az]
                end
            end
        end
    end

    rho ./= voxelvol
    return rho
end

# --------------------------------
# Gather potential back to particles
# --------------------------------
function _gather_bspline4(
    field,
    pos,
    box_inv
)
    Nx, Ny, Nz = size(field)
    N = size(pos, 2)
    vals = zeros(eltype(field), N)

    for i in 1:N
        r = @view pos[:, i]
        s = box_inv * r

        sx = _wrap01(s[1]) * Nx
        sy = _wrap01(s[2]) * Ny
        sz = _wrap01(s[3]) * Nz

        ix = floor(Int, sx)
        iy = floor(Int, sy)
        iz = floor(Int, sz)

        tx = sx - ix
        ty = sy - iy
        tz = sz - iz

        wx = _bspline4_weights(tx)
        wy = _bspline4_weights(ty)
        wz = _bspline4_weights(tz)

        acc = 0.0
        for (ax, ox) in enumerate((-1, 0, 1, 2))
            gx = mod(ix + ox, Nx) + 1
            for (ay, oy) in enumerate((-1, 0, 1, 2))
                gy = mod(iy + oy, Ny) + 1
                for (az, oz) in enumerate((-1, 0, 1, 2))
                    gz = mod(iz + oz, Nz) + 1
                    acc += wx[ax] * wy[ay] * wz[az] * field[gx, gy, gz]
                end
            end
        end

        vals[i] = acc
    end

    return vals
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
    mesh
)
    Nx, Ny, Nz = mesh
    V = abs(det(box))

    L1 = norm(box[:,1])
    L2 = norm(box[:,2])
    L3 = norm(box[:,3])

    @show L1 / Nx
    @show L2 / Ny
    @show L3 / Nz

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

        nvec = SVector_like(ni, nj, nk)  # helper defined below
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

# tiny helper to avoid StaticArrays dependency
@inline SVector_like(x, y, z) = [x, y, z]

# --------------------------------
# Main function
# --------------------------------
function pme_bspline(
    pos::Matrix{Float64},
    q::Vector{Float64};
    box::Matrix{Float64},
    rcut::Float64,
    alpha=nothing,
    alpha_factor::Float64=4.0,
    mesh::NTuple{3,Int}=(32, 32, 32),
    include_short::Bool=true,
    include_long::Bool=true,
    subtract_self::Bool=true,
    subtract_mean_potential::Bool=false
)
    N = size(pos, 2)

    @assert size(pos, 1) == 3 "positions must be 3×N"
    @assert size(box) == (3, 3) "box must be a 3×3 lattice matrix with lattice vectors as columns"
    @assert length(q) == N "length(q) must equal number of atoms"
    @assert abs(sum(q)) < 1e-8 "Periodic PME requires a neutral cell unless you add a neutralizing background"
    @assert rcut > 0.0
    @assert abs(det(box)) > 1e-12 "box must be invertible"

    box_inv = inv(box)
    α = isnothing(alpha) ? (alpha_factor / rcut) : alpha

    phi_s = zeros(Float64, N)
    phi_l = zeros(Float64, N)

    Es = include_short ? ke * _ewald_real!(phi_s, pos, q, box, box_inv, α, rcut) : 0.0
    El = include_long  ? ke * _recip!(phi_l, pos, q, box, box_inv, α, mesh)      : 0.0

    phi = ke .* (phi_s .+ phi_l)

    Eself = 0.0
    if subtract_self
        c = ke * α / sqrt(pi)
        Eself = -c * sum(q .^ 2)
        phi .-= 2c .* q
    end

    if subtract_mean_potential
        φ̄ = sum(phi) / N
        phi .-= φ̄
        phi_s .-= sum(phi_s) / N
        phi_l .-= sum(phi_l) / N
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

function brute_force_periodic_coulomb(
    pos::Matrix{Float64},
    q::Vector{Float64};
    box::Matrix{Float64},
    nimg::Int=3,
    subtract_mean_potential::Bool=false
)
    N = size(pos, 2)

    @assert size(pos, 1) == 3 "positions must be 3×N"
    @assert size(box) == (3, 3) "box must be 3×3"
    @assert length(q) == N "charges must have length N"

    phi = zeros(Float64, N)
    E = 0.0

    # image vectors n = (nx, ny, nz), with shift = box * n
    for i in 1:N
        ri = @view pos[:, i]

        for j in 1:N
            rj = @view pos[:, j]
            qj = q[j]

            for nx in -nimg:nimg, ny in -nimg:nimg, nz in -nimg:nimg
                # skip exact self interaction in the home cell
                if i == j && nx == 0 && ny == 0 && nz == 0
                    continue
                end

                shift = box * [nx, ny, nz]
                dr = ri - (rj + shift)
                r = norm(dr)

                phi[i] += ke * qj / r
            end
        end
    end

    # total energy = 1/2 sum_i q_i phi_i
    E = 0.5 * sum(q .* phi)

    if subtract_mean_potential
        phi .-= mean(phi)
    end

    return (
        potentials = phi,
        energy = E,
        nimg = nimg,
    )
end

using LinearAlgebra

function zahn_potential(
    pos::Matrix{Float64},
    q::Vector{Float64};
    box::Matrix{Float64},
    rcut::Float64,
    alpha::Union{Nothing,Float64}=nothing,
    alpha_factor::Float64=4.0,
    ke::Float64=14.3996454784255,   # eV*Å
    subtract_mean_potential::Bool=false
)
    N = size(pos, 2)

    @assert size(pos, 1) == 3 "positions must be 3×N"
    @assert size(box) == (3, 3) "box must be 3×3 with lattice vectors as columns"
    @assert length(q) == N "charges must have length N"
    @assert rcut > 0.0
    @assert abs(det(box)) > 1e-12 "box must be invertible"

    box_inv = inv(box)
    α = isnothing(alpha) ? (alpha_factor / rcut) : alpha

    # For nearest-image only, this is safest if rcut is not too large.
    a1 = norm(box[:,1])
    a2 = norm(box[:,2])
    a3 = norm(box[:,3])
    @assert rcut <= min(a1, a2, a3)/2 "Use rcut <= half the shortest box-vector length for this minimum-image implementation"

    phi = zeros(Float64, N)
    E = 0.0

    # Shift term so V(rcut)=0
    vshift = erfc(α * rcut) / rcut

    for i in 1:N-1
        ri = @view pos[:, i]
        qi = q[i]

        for j in i+1:N
            rj = @view pos[:, j]
            qj = q[j]

            # minimum-image displacement in a general cell
            ds = box_inv * (ri - rj)
            ds .-= round.(ds)
            dr = box * ds
            r = norm(dr)

            if r < rcut && r > 1e-14
                v = ke * (erfc(α * r) / r - vshift)

                # pair energy
                E += qi * qj * v

                # site potentials, defined so E = 1/2 sum_i q_i phi_i
                phi[i] += qj * v
                phi[j] += qi * v
            end
        end
    end

    if subtract_mean_potential
        phi .-= mean(phi)
    end

    return (
        alpha = α,
        rcut = rcut,
        potentials = phi,
        energy = E,
    )
end

using Hamster, Plots, Plots.Measures, Statistics, HDF5, MultivariateStats, StatsBase, Ewalder, FiniteDiff

blue = RGB(0.0, 0.45, 0.70)
orange = RGB(0.90, 0.60, 0.0)
green = RGB(0.0, 0.60, 0.50)
yellow = RGB(0.95, 0.90, 0.25)
purple = RGB(0.80, 0.60, 0.70)
sky_blue = RGB(0.35, 0.70, 0.90)
vermilion = RGB(0.80, 0.40, 0.0)
black = RGB(0.0, 0.0, 0.0)
colors = [blue, orange, green, yellow, purple, sky_blue, vermilion, black]

labelsize = 11
ticksize = ceil(Int64, labelsize / 1.25)
@show labelsize
@show ticksize
@show labelsize / ticksize

path = "/home/martin/Documents/Doktorarbeit/Projects/physics_informed_hamster/02_CsPbBr3/02_Hamster_TB_PC_fit/02_tb_fit"

potentials = Dict{String, Vector{Float64}}("Pb"=>[], "Br"=>[], "Cs"=>[])
for i in 1:1
    path2 = "/home/martin/Documents/Doktorarbeit/Projects/physics_informed_hamster/02_CsPbBr3/01_DFT_data/05_md_data/2x2x2_DFT_train_val_test_sets/425K_soc_test/config_$i"
    poscar = Hamster.read_poscar(joinpath(path2, "POSCAR"))
    pos = Hamster.frac_to_cart(poscar.rs_atom, poscar.lattice)
    s = 1
    charges = Dict{String, Float64}("Pb"=>2*s, "Cs"=>s, "Br"=>-s)
    atom_types = Hamster.number_to_element.(poscar.atom_types)
    q = [charges[type] for type in atom_types]

    box = poscar.lattice

    res = pme_bspline(
        pos,
        q;
        box=box,
        rcut=5.5,
        mesh=(32, 32, 32),
        include_short=true,
        include_long=true,
        subtract_self=true,
        subtract_mean_potential=false
    )

    @show fd_relative_potentials(x->pme_bspline(
        pos,
        x;
        box=box,
        rcut=5.5,
        mesh=(32, 32, 32),
        include_short=true,
        include_long=true,
        subtract_self=true,
        subtract_mean_potential=false
    ).energy, q)

    pos = Vector{Float64}[r for r in eachcol(pos)]
    latvecs = [v for v in eachcol(box)]
    @show typeof(pos)
    sys = Ewalder.System(; latvecs=latvecs, pos=pos)
    E = Ewalder.energy(sys; charges=q)
    @show E
    @show E * ke

    #@show fd_relative_potentials(x->ke*Ewalder.energy(sys, charges=x), q)

    s = 1
    charges = Dict{String, Float64}("Pb"=>2*s, "Cs"=>s, "Br"=>-s)
    atom_types = Hamster.number_to_element.(poscar.atom_types)
    q2 = [charges[type] for type in atom_types]
    @show ke*Ewalder.energy(sys, charges=q2)

    #println("alpha = ", res.alpha)
    println("E_total = ", res.energy)
    #println("ϕ = ", res.potentials)

    #ref = brute_force_periodic_coulomb(
    #    pos,
    #    q;
    #    box=box,
    #    nimg=7,
    #)

    #println("Reference energy = ", ref.energy)
    #println("Residual potentials = ", ref.potentials .- res.potentials)
    #println("Mean ϕ = ", std([res.potentials[i] for i in eachindex(res.potentials) if atom_types[i] == "Br"]))
    append!(potentials["Pb"], [res.potentials[i] for i in eachindex(res.potentials) if atom_types[i] == "Pb"])
    append!(potentials["Br"], [res.potentials[i] for i in eachindex(res.potentials) if atom_types[i] == "Br"])
    append!(potentials["Cs"], [res.potentials[i] for i in eachindex(res.potentials) if atom_types[i] == "Cs"])
end

function finite_difference_potentials(energy_fn, q; δ=1e-5)
    N = length(q)
    φ = zeros(Float64, N)

    for i in 1:N
        qp = copy(q)
        qm = copy(q)

        qp[i] += δ
        qm[i] -= δ

        # preserve neutrality by compensating uniformly on all other atoms
        for j in 1:N
            if j != i
                qp[j] -= δ / (N - 1)
                qm[j] += δ / (N - 1)
            end
        end

        Ep = energy_fn(qp)
        Em = energy_fn(qm)

        φ[i] = (Ep - Em) / (2δ)
    end

    return φ
end

function fd_relative_potentials(energy_fn, q; δ=1e-5, ref=1)
    N = length(q)
    φrel = zeros(Float64, N)

    for i in 1:N
        i == ref && continue

        qp = copy(q)
        qm = copy(q)

        qp[i] += δ
        qp[ref] -= δ

        qm[i] -= δ
        qm[ref] += δ

        φrel[i] = (energy_fn(qp) - energy_fn(qm)) / (2δ)
    end

    return φrel
end

ps = map(enumerate(["Cs", "Pb", "Br"])) do (i, type)
    p = plot(framestyle=:box, guidefontsize=labelsize, tickfontsize=ticksize, legendfontsize=labelsize)      
    hist = fit(Histogram, potentials[type] .- mean(potentials[type]), nbins=100)
    xs = hist.edges[1]
    ys = hist.weights ./ sum(hist.weights)
    plot!(xs[2:end], ys, label=type, lw=2, color=colors[i])
    xlims!(-2, 2)
    return p
end

fig = plot(ps..., layout=(3, 1), title="Ewald full")
#savefig("ewald_onsite_correction_long_range.pdf")

e = 1.602176634e-19          # C
eps0 = 8.8541878128e-12      # F/m
eV = 1.602176634e-19         # J
angstrom = 1e-10             # m

k_e = e^2 / (4π * eps0) / eV / angstrom
println(k_e)   # 14.399645478...

for n in [4, 8, 16]
    path2 = "/home/martin/Downloads"
    poscar = Hamster.read_poscar(joinpath(path2, "POSCAR_$n$n$n"))
    pos = Hamster.frac_to_cart(poscar.rs_atom, poscar.lattice)
    s = 1
    charges = Dict{String, Float64}("Pb"=>2*s, "Cs"=>s, "Br"=>-s)
    atom_types = Hamster.number_to_element.(poscar.atom_types)
    q = [charges[type] for type in atom_types]

    box = poscar.lattice

    res = pme_bspline(
        pos,
        q;
        box=box,
        rcut=5.5,
        mesh=(256, 256, 256),
        include_short=true,
        include_long=true,
        subtract_self=true,
        subtract_mean_potential=false
    )

    println("alpha = ", res.alpha)
    println("E_total = ", res.energy)
    #println("ϕ = ", res.potentials)
    println("Mean ϕ = ", mean([res.potentials[i] for i in eachindex(res.potentials) if atom_types[i] == "Br"]))
    println("std ϕ = ", std([res.potentials[i] for i in eachindex(res.potentials) if atom_types[i] == "Br"]))
end