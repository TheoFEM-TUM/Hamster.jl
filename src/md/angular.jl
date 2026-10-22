# Angular parts of the TB matrix elements as polynomials of the bond direction.
#
# The Slater-Koster angular factor of a matrix element between orbitals with angular momenta l₁ and l₂ is a polynomial
# of degree l₁ + l₂ in the direction cosines of the bond. Instead of differentiating the angles, frames and sign
# conventions of `geometry_entries!`, its values are sampled for random directions and fitted exactly by polynomials,
# whose derivatives are simple. The fit residual is checked, so any angular function that is not such a polynomial
# (which would be a discontinuity of the Hamiltonian) is detected.

"""
    monomial_exponents(D)

Exponents (a, b, c) of the monomials xᵃ yᵇ zᶜ of degree ≤ `D`.
"""
monomial_exponents(D) = [SVector{3, Int}(a, b, t - a - b) for t in 0:D for a in t:-1:0 for b in (t - a):-1:0]

"""
    monomials!(m, ∇m, exps, n)

Values `m[k]` and gradients `∇m[k]` of the monomials with exponents `exps` at `n`.
"""
@inline function monomials!(m, ∇m, exps, n)
    x, y, z = n
    @inbounds for (k, e) in enumerate(exps)
        a, b, c = e
        px = a > 0 ? x^(a - 1) : 0.0; py = b > 0 ? y^(b - 1) : 0.0; pz = c > 0 ? z^(c - 1) : 0.0
        xa = x^a; yb = y^b; zc = z^c
        m[k] = xa * yb * zc
        ∇m[k] = SVector{3, Float64}(a * px * yb * zc, b * xa * py * zc, c * xa * yb * pz)
    end
    return nothing
end

"""
    random_directions(Ns; seed=1)

`Ns` random unit vectors, away from the coordinate axes (where some bond frames are degenerate). Their z components
are multiples of 0.001: the bond frame of the geometry tensor (`get_transformed_system`) uses the angle to the z axis
from the cosine rounded to five digits (`calc_angle`), which is exact for these directions, so the sampled angular
parts are the exact Slater-Koster factors.
"""
function random_directions(Ns; seed=1)
    rng = Xoshiro(seed)
    dirs = SVector{3, Float64}[]
    while length(dirs) < Ns
        z = round(1.9 * rand(rng) - 0.95, digits=3)
        φ = 2π * rand(rng)
        ρ = sqrt(1 - z^2)
        v = SVector{3, Float64}(ρ * cos(φ), ρ * sin(φ), z)
        minimum(abs, v) > 0.05 && round(z, digits=5) == z && push!(dirs, v)
    end
    return dirs
end

"""
    AngularTerm

One contribution `V[v] radial_k(r) Σ coeffs[m] monomial_m(r̂)` to the matrix element between orbital `a` of the first
and orbital `b` of the second ion of a pair, with the parameter already multiplied into `coeffs`.
"""
struct AngularTerm
    a :: Int32
    b :: Int32
    k :: Int32
    coeffs :: Vector{Float64}
end

"""
    AngularClass

All terms of the matrix elements between the orbitals of two ion types, for one ordering of the two ions (which can
change the sign convention for ions of the same type), with the monomial exponents of the polynomials.
"""
struct AngularClass
    terms :: Vector{AngularTerm}
    exps :: Vector{SVector{3, Int}}
    ks :: Vector{Int32}
end

"""
    fit_angular_class(ctx, iion1, iion2, offset1, offset2, params, nnlabel; Ns=400, tol=1e-9)

Polynomials of the angular parts of the matrix elements between the ions `iion1` and `iion2` of the reference structure
of `ctx` (a `GeometryContext`), evaluated by `geometry_entries!` with the orbital offsets `offset1` and `offset2` (which
decide the ordering conventions) for `Ns` random bond directions and fitted by least squares. Errors if the residual of
a term exceeds `tol` (relative to its largest value).
"""
function fit_angular_class(ctx, iion1, iion2, offset1, offset2, params, nnlabel; Ns=400, tol=1e-9)
    lmax(ls) = maximum(l -> l < 0 ? 2 : l, ls; init=0)
    D = lmax(ctx.orbital_ls[iion1]) + lmax(ctx.orbital_ls[iion2])
    exps = monomial_exponents(D)
    dirs = random_directions(Ns)
    radial = ones(length(ctx.basis.overlaps))
    angles = zeros(5)
    # Values of every (a, b, k, v) term for all directions
    values = Dict{NTuple{4, Int}, Vector{Float64}}()
    r⃗₁ = SVector{3, Float64}(0, 0, 0)
    for (s, n̂) in enumerate(dirs)
        geometry_entries!(ctx, iion1, iion2, offset1, offset2, r⃗₁, n̂, nnlabel, radial, angles) do v, i, j, k, hval
            key = (i - offset1, j - offset2, k, v)
            vals = get!(() -> zeros(Ns), values, key)
            vals[s] += hval
        end
    end
    A = zeros(Ns, length(exps))
    m = zeros(length(exps)); ∇m = zeros(SVector{3, Float64}, length(exps))
    for (s, n̂) in enumerate(dirs)
        monomials!(m, ∇m, exps, n̂)
        A[s, :] .= m
    end
    Apinv = pinv(A, rtol=1e-12)
    # Terms of the same orbitals and overlap are combined (their parameters are known)
    combined = Dict{NTuple{3, Int}, Vector{Float64}}()
    for ((a, b, k, v), vals) in values
        v == 0 && continue
        c = Apinv * vals
        scale = maximum(abs, vals)
        if scale > 0 && maximum(abs, A * c - vals) > tol * max(scale, 1e-3)
            error("The angular part of a TB matrix element (orbitals $a and $b of ion types $(ctx.ion_types[iion1]) and " *
                  "$(ctx.ion_types[iion2]), overlap $(string(ctx.basis.overlaps[k]))) is not a polynomial of the bond " *
                  "direction (residual $(maximum(abs, A * c - vals))). The MD needs smooth matrix elements.")
        end
        acc = get!(() -> zeros(length(exps)), combined, (a, b, k))
        acc .+= params[v] .* c
    end
    terms = [AngularTerm(a, b, k, c) for ((a, b, k), c) in combined if maximum(abs, c; init=0.0) > 0]
    sort!(terms, by=t -> (t.a, t.b, t.k))
    return AngularClass(terms, exps, unique(Int32[t.k for t in terms]))
end
