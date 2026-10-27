# Long-range electrostatics of slabs (cells with vacuum along one lattice vector, `slab_axis` of the Ewald block).
#
# Wolf and Zahn keep only the short-range part erfc(αr)/r of the Coulomb interaction. In a slab, the long-range part that
# matters is its average over the planes parallel to the surface (the k_∥ = 0 Fourier component): aligned dipoles of
# successive layers add up to a potential step across the slab, which a truncated sum cannot see. For an isolated slab
# (vacuum on both sides), the plane-averaged potential of the smooth part erf(αr)/r of point charges q_j at heights z_j is
# the potential of Gaussian charge sheets,
#   φ̄(z) = -(2π/A) ke Σ_j q_j [u erf(αu) + exp(-α²u²)/(α√π)],   u = z - z_j,
# with the area A of the cell in the plane of the slab. Added to the Wolf/Zahn potentials with the same α, it completes
# them by the plane-averaged long range without double counting. The isolated-slab form has no field in the vacuum,
# i.e. it includes the dipole correction. For the full Ewald sum (method = ewald), which already contains the long range
# but treats the slab as periodic along the axis, `slab_axis` adds the dipole correction (Yeh and Berkowitz) instead:
#   φ_i += 4π ke M z_i / V,   M = Σ_j q_j z_j.
# The charges are made neutral with a uniform background on the atoms (q_j - mean(q)), so that the potential of the
# slab stays finite in the vacuum also for effective charges that do not add up to zero. Where the net charge is
# compensated instead (`charge_compensation`), see the second part of this file.

"""
    slab_heights(pos, lattice, axis)

Heights of the atoms along the normal of the plane spanned by the two other lattice vectors, measured from the vacuum:
the fractional coordinates along `axis` are shifted so that the largest gap between atoms (the vacuum) lies at the cell
boundary, i.e. the slab is not cut by the periodic boundary. Returns the heights, the area of the cell in the plane and
the width of the vacuum.
"""
function slab_heights(pos, lattice, axis)
    a, b = [i for i in 1:3 if i ≠ axis]
    A = norm(cross(lattice[:, a], lattice[:, b]))
    h = abs(det(lattice)) / A                          # height of the cell along the normal
    lattice_inv = inv(lattice)
    f = [mod((lattice_inv * r)[axis], 1.0) for r in pos]
    fs = sort(f)
    gaps = vcat(diff(fs), fs[1] + 1 - fs[end])         # gap after each sorted atom (the last one across the boundary)
    k = argmax(gaps)
    start = k == length(fs) ? fs[1] : fs[k+1]          # the slab starts above the largest gap
    z = [mod(fi - start, 1.0) * h for fi in f]
    return z, A, gaps[k] * h
end

"""
    slab_potentials(pos, q, lattice, axis; method, alpha)

Long-range electrostatic potentials (eV per elementary charge) of the atoms of a slab with vacuum along the lattice
vector `axis`, to be added to the potentials of `ewald_sum`: for `wolf` and `zahn` the plane-averaged potential of the
smooth part erf(αr)/r of the charges (Gaussian charge sheets of an isolated slab), for `ewald` the dipole correction.
The charges are neutralized by a uniform background on the atoms. See the header of `ewald_slab.jl`.
"""
function slab_potentials(pos, q, lattice, axis; method, alpha)
    (1 ≤ axis ≤ 3) || error("slab_axis = $axis: the lattice vector across the vacuum is 1, 2 or 3 (0: no slab).")
    z, A, vacuum = slab_heights(pos, lattice, axis)
    qn = q .- sum(q) / length(q)
    if lowercase(method) == "ewald"
        V = abs(det(lattice))
        M = sum(qn .* z)
        return ke * 4π * M / V .* (z .- sum(z) / length(z))
    end
    α = alpha
    φ = zeros(length(z))
    tforeach(eachindex(z)) do i
        s = 0.0
        @inbounds for j in eachindex(z)
            u = z[i] - z[j]
            s += qn[j] * (u * erf(α * u) + exp(-(α * u)^2) / (α * sqrt(π)))
        end
        φ[i] = -(2π / A) * ke * s
    end
    return φ
end

# Compensation of the net effective charge Q = Σ q of a structure (`charge_compensation` of the Ewald block). Effective
# charges need not add up to zero (non-stoichiometric slabs, vacancies), and where the compensating charge -Q sits
# matters for the long range of a slab: spread over all atoms (`global`, the uniform background of `slab_potentials`),
# the layers of a slab whose effective charges do not cancel (e.g. ±σ alternating, PbBr2-terminated on both sides)
# leave the surfaces charged, with a potential step of volts between surface and interior. The electrons compensate
# such charges at the surfaces (`surface`), so that no macroscopic field remains inside, or next to a defect (`local`).

"""
    slab_layers(z; layer_gap=1.0)

Atomic layers of a slab from the heights `z` of the atoms: a new layer starts where the gap to the next higher atom
exceeds `layer_gap` (Å). Returns the layer index of every atom, counted from the lowest layer.
"""
function slab_layers(z; layer_gap=1.0)
    o = sortperm(z); layer = zeros(Int, length(z)); l = 1
    for k in eachindex(o)
        k > 1 && z[o[k]] - z[o[k-1]] > layer_gap && (l += 1)
        layer[o[k]] = l
    end
    return layer
end

"""
    surface_compensation(z, q; layer=slab_layers(z))

Compensating charges of a slab with heights `z` and charges `q`: the net charge -Σq is placed on the lowest and the
highest layer (equal per atom of the layer) and split between them so that the plane-averaged field between the outermost
layers vanishes on average (gaps between the layers weighted by their width, the two outermost gaps by half). For a
periodic stacking this is exact: e.g. half of the layer charge on each side of a symmetric slab with layers of charge
±σ, and σ a / (a + b) for a polar stacking with alternating spacings a and b (the surface charge of the bulk
polarization).
"""
function surface_compensation(z, q; layer=slab_layers(z))
    nl = maximum(layer)
    nl ≥ 3 || error("charge_compensation = surface: the slab has $nl atomic layer(s) along slab_axis, at least 3 are needed.")
    σ = zeros(nl); h = zeros(nl); n = zeros(Int, nl)
    for i in eachindex(q)
        σ[layer[i]] += q[i]; h[layer[i]] += z[i]; n[layer[i]] += 1
    end
    h ./= n
    # the field in the gap above layer k is ∝ the charge up to layer k (S_k + t with the compensation t on the lowest layer)
    S = cumsum(σ)[1:nl-1]
    w = diff(h); w[1] /= 2; w[end] /= 2
    t = -dot(w, S) / sum(w)
    c = zeros(length(q))
    lo = findall(==(1), layer); hi = findall(==(nl), layer)
    c[lo] .= t / length(lo)
    c[hi] .= (-sum(q) - t) / length(hi)
    return c
end

"""
    coordination_numbers(pos, types, q, lattice; shell_gap=1.1)

Number of nearest neighbors of opposite charge (`q`) of every atom at the positions `pos` in the periodic cell
`lattice` (columns: lattice vectors). The first shell of every pair of species ends at the first jump to a distance more
than `shell_gap` times the previous one, among the distances up to twice the shortest one (all of them if there is no
jump).
"""
function coordination_numbers(pos, types, q, lattice; shell_gap=1.1)
    N = length(pos)
    pairs = [(i, j) for i in 1:N, j in 1:N if q[i] * q[j] < 0]
    isempty(pairs) && return zeros(Int, N)
    # images of the cell that can hold a neighbor within twice the shortest distance of opposite charges
    minimum_image(d) = d - lattice * round.(lattice \ d)
    dmin = minimum(norm(minimum_image(pos[j] - pos[i])) for (i, j) in pairs)
    h = [abs(det(lattice)) / norm(cross(lattice[:, mod1(d + 1, 3)], lattice[:, mod1(d + 2, 3)])) for d in 1:3]
    nimg = [ceil(Int, 2dmin / h[d]) + 1 for d in 1:3]
    images = [lattice * [n1, n2, n3] for n1 in -nimg[1]:nimg[1], n2 in -nimg[2]:nimg[2], n3 in -nimg[3]:nimg[3]]
    dists = Dict{Tuple{UInt8, UInt8}, Vector{Float64}}()
    neighbors = [Tuple{Int, UInt8, Float64}[] for _ in 1:N]
    for (i, j) in pairs, T in images
        r = norm(pos[j] + T - pos[i])
        if r ≤ 2dmin
            push!(neighbors[i], (j, types[j], r)); push!(get!(dists, (types[i], types[j]), Float64[]), r)
        end
    end
    cutoff = Dict{Tuple{UInt8, UInt8}, Float64}()
    for (key, d) in dists
        sort!(d); cutoff[key] = d[end]
        for k in 2:length(d)
            if d[k] > shell_gap * d[k-1]
                cutoff[key] = (d[k-1] + d[k]) / 2; break
            end
        end
    end
    return [count(((j, t, r),) -> r ≤ cutoff[(types[i], t)], neighbors[i]) for i in 1:N]
end

"""
    local_compensation(pos, types, q, lattice)

Compensating charges at defects: the net charge -Σq is distributed over the atoms with fewer neighbors of opposite
charge than the most common number of their species (`coordination_numbers`), in proportion to the missing neighbors,
e.g. on the cations next to an anion vacancy.
"""
function local_compensation(pos, types, q, lattice)
    Q = sum(q)
    abs(Q) < 1e-10 && return zeros(length(q))
    cn = coordination_numbers(pos, types, q, lattice)
    missing_nb = zeros(length(q))
    for t in unique(types)
        m = findall(==(t), types)
        counts = Dict{Int, Int}()
        foreach(i -> counts[cn[i]] = get(counts, cn[i], 0) + 1, m)
        bulk = first(argmax(last, collect(counts)))
        missing_nb[m] .= max.(bulk .- cn[m], 0)
    end
    sum(missing_nb) > 0 || error("charge_compensation = local: no atom with missing neighbors found to compensate the net " *
                                 "charge $(round(Q, digits=4)) e; use charge_compensation = global.")
    return -Q .* missing_nb ./ sum(missing_nb)
end

"""
    compensation_charges(mode, pos, types, q, lattice; slab_axis=0)

Charges to add to the effective charges `q` for `charge_compensation = mode`: none for `global` (the uniform background
of the slab terms), `surface_compensation` on the outermost layers along `slab_axis` for `surface`, and
`local_compensation` for `local`.
"""
function compensation_charges(mode, pos, types, q, lattice; slab_axis=0)
    if mode == "global"
        return zeros(length(q))
    elseif mode == "surface"
        slab_axis > 0 || error("charge_compensation = surface needs slab_axis (the lattice vector across the vacuum).")
        z, _, _ = slab_heights(pos, lattice, slab_axis)
        return surface_compensation(z, q)
    elseif mode == "local"
        return local_compensation(pos, types, q, lattice)
    end
    error("charge_compensation = $mode: unknown, use global, surface or local.")
end
