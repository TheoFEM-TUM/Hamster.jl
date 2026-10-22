# Units of the MD: Å, fs, eV, amu, K.

"Boltzmann constant (eV/K)."
const KB_EV = 8.617333262e-5

"Acceleration (Å/fs²) of a force of 1 eV/Å on a mass of 1 amu."
const EV_PER_AMU = 9.648533212331e-3

"Pressure (GPa) of an energy density of 1 eV/Å³."
const EV_PER_A3_IN_GPA = 160.21766208

"""
    atomic_mass(type)

Mass (amu) of the element with proton number `type`.
"""
atomic_mass(type) = Float64(strip_unit(elements[Int(type)].atomic_mass))
strip_unit(x::Number) = x
strip_unit(x) = x.val

"""
    log1pexp(x)

`log(1 + exp(x))` without overflow.
"""
log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    fermi(ε, μ, kT)

Fermi-Dirac occupation of a state at `ε`.
"""
function fermi(ε, μ, kT)
    x = (ε - μ) / kT
    return x > 0 ? exp(-x) / (1 + exp(-x)) : 1 / (1 + exp(x))
end

"""
    grand_potential(ε, μ, kT)

Grand potential `-kT log(1 + exp(-(ε - μ)/kT))` of a state at `ε`, whose derivative with respect to `ε` is the
occupation `fermi(ε, μ, kT)`. The Mermin free energy of `N` electrons is `Σ grand_potential + μ N`.
"""
grand_potential(ε, μ, kT) = -kT * log1pexp(-(ε - μ) / kT)

"""
    chebyshev_coefficients(f, M, a, b; Nq=max(2M, 512))

Coefficients `c[n+1]` of the expansion `f(ε) ≈ Σₙ cₙ Tₙ((ε - b)/a)` on `[b - a, b + a]` with `M` terms, from the
Chebyshev-Gauss quadrature with `Nq` nodes.
"""
function chebyshev_coefficients(f, M, a, b; Nq=max(2M, 512))
    θs = [π * (j - 0.5) / Nq for j in 1:Nq]
    fs = [f(a * cos(θ) + b) for θ in θs]
    c = zeros(M)
    for n in 0:M-1
        s = 0.0
        @inbounds for j in 1:Nq
            s += fs[j] * cos(n * θs[j])
        end
        c[n+1] = (n == 0 ? 1 : 2) * s / Nq
    end
    return c
end

"""
    chebyshev_derivative(c, a)

Coefficients of the derivative `d/dε Σₙ cₙ Tₙ((ε - b)/a)` of a Chebyshev series, which is again a series of `length(c)`
terms (the last one zero). Used for the density matrix, the exact derivative of the polynomial grand potential, so that
energies and forces are consistent for any number of terms.
"""
function chebyshev_derivative(c, a)
    M = length(c)
    d = zeros(M + 1)
    for n in M-1:-1:1
        d[n] = d[n+2] + 2n * c[n+1]
    end
    d[1] /= 2
    return d[1:M] ./ a
end

"""
    chebyshev_series(c, x)

Value of `Σₙ cₙ Tₙ(x)` (Clenshaw recursion).
"""
function chebyshev_series(c, x)
    b1 = 0.0; b2 = 0.0
    for n in length(c):-1:2
        b1, b2 = 2x * b1 - b2 + c[n], b1
    end
    return x * b1 - b2 + c[1]
end

"""
    bisect(f, lo, hi; tol=1e-12, maxiter=200)

Root of the increasing function `f` in `[lo, hi]` by bisection (the interval is widened if needed).
"""
function bisect(f, lo, hi; tol=1e-12, maxiter=200)
    for _ in 1:60
        f(lo) ≤ 0 && break
        lo -= (hi - lo)
    end
    for _ in 1:60
        f(hi) ≥ 0 && break
        hi += (hi - lo)
    end
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
        hi - lo < tol && break
    end
    return (lo + hi) / 2
end

"""
    min_image(Δ, L, Linv)

Shortest periodic image of the Cartesian difference vector `Δ` in the cell with lattice vectors `L` (columns).
"""
function min_image(Δ, L, Linv)
    s = Linv * Δ
    s = s .- round.(s)
    return L * s
end
