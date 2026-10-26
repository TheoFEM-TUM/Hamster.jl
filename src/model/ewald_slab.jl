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
# slab stays finite in the vacuum also for effective charges that do not add up to zero.

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
