"""
    str_to_orb(str::String)

Convert an orbital name given as a string to its corresponding orbital object.

# Arguments
- `str::String`: The name of the orbital as a string. Valid strings include `"s"`, `"px"`, `"py"`, `"pz"`, `"dz2"`, `"dxy"`, `"dxz"`, `"dyz"`, `"dx2_y2"`, `"sp3"`, `"sp3dr2"`, `"pxdx2"`, `"pydy2"`, and `"pzdz2"`.

# Returns
- `Orbital`: The corresponding orbital object based on the input string.
"""
function str_to_orb(str)::Angular
    orbdict = Dict("s"=>s(), "px"=>px(), "py"=>py(), "pz"=>pz(),
        "dz2"=>dz2(), "dxy"=>dxy(), "dxz"=>dxz(), "dyz"=>dyz(), 
        "dx2_y2"=>dx2_y2(), "sp3"=>sp3(), "sp3dr2"=>sp3dr2(), "pxdx2"=>pxdx2(), "pydy2"=>pydy2(), "pzdz2"=>pzdz2())
    return orbdict[str]
end

"""
    get_spherical(l, m)

Returns the spherical harmonic function for a given orbital angular momentum quantum number `l` and magnetic quantum number `m`.

# Arguments
- `l::Int`: The orbital angular momentum quantum number (`l = 0` for `s`, `l = 1` for `p`, `l = 2` for `d`, etc.).
- `m::Int`: The magnetic quantum number, which ranges from `-l` to `l`.

# Returns
- The corresponding spherical harmonic function as a vector, based on the values of `l` and `m`.
"""
get_spherical(l, m)::Angular = [[s()], [pz(), px()], [dz2(), dxz(), dx2_y2()]][l+1][m+1]


# Angular parts produce 0/0 for r = [0, 0, 0]
nan_to_zero(x) = isnan(x) ? zero(x) : x

"""
    (orb::OrbitalFunction)(r⃗)

Evaluate the orbital `orb` at the point `r⃗`, which can be one- or
two-dimensional or a tuple including the norm.
"""
abstract type OrbitalFunction <: Function end
function (orb::OrbitalFunction)(r⃗::AbstractArray{Float64, 2})
    N = size(r⃗)[2]; ϕ = zeros(Float64, N)
    for i in 1:N
        r⃗ᵢ = @view r⃗[:, i]
        ϕ[i] = orb(r⃗ᵢ)
    end
    return ϕ
end
function (orb::OrbitalFunction)(r⃗::AbstractArray{Float64, 1})
    x, y, z = r⃗
    ϕ = orb(x, y, z)
    return nan_to_zero(ϕ)
end
function (orb::OrbitalFunction)(p::Tuple)::Float64
    x, y, z, r = p
    ϕ = orb(x, y, z, r)
    return nan_to_zero(ϕ)
end
function (orb::OrbitalFunction)(r⃗::A1, r⃗₀::A2) where {A1,A2<:AbstractArray{Float64, 1}}
    x, y, z = r⃗; x₀, y₀, z₀ = r⃗₀
    ϕ = orb(x - x₀, y - y₀, z - z₀)
    ϕ = nan_to_zero(ϕ)
    return ϕ
end 

"""
    Radial

Radial parts for atomic s orbitals with n=1,2,3,4,5,6.
"""
abstract type Radial<:OrbitalFunction end
(f::Radial)(α::Int64) = f(Float64(α))

# n = 1
struct R1<:Radial
    α :: Float64
    n :: Int64
end
R1(α::Float64) = R1(α, 1)
function(f::R1)(x::F, y::F, z::F) where {F<:Float64}
    r = √(x^2 + y^2 + z^2)
    α = f.α
    Rᵣ = 2*α^(3/2) * exp(-α*r)
    return Rᵣ
end
(f::R1)(x, y, z, r) = 2*f.α^(3/2) * exp(-f.α*r)

# n = 2
struct R2<:Radial
    α :: Float64
    n :: Int64
end
R2(α::Float64) = R2(α, 2)
function(f::R2)(x::T, y::T, z::T) where {T<:Float64}
    r = √(x^2 + y^2 + z^2)
    α = f.α
    Rᵣ = 1/(√8) * α^(3/2) * (2 - α*r) * exp(-α*r/2)
    return Rᵣ
end
(f::R2)(x, y, z, r) = 1/(√8) * f.α^(3/2) * (2 - f.α*r) * exp(-f.α*r/2)

# n = 3
struct R3{A,B,C,D<:Float64} <:Radial
    α :: A
    c1 :: B
    c2 :: C
    c3 :: D
    n :: Int64
end
R3(α::Float64) = R3(α, √(4/27)*α^(3/2), 2/3 *α, 2/27*α^2, 3)
function(f::R3)(x::T, y::T, z::T) where {T<:Float64}
    r = √(x^2 + y^2 + z^2)
    α = f.α
    Rᵣ = f.c1*(1-(f.c2*r)+(f.c3*r^2))*exp(-α*r/3)
    return Rᵣ
end
(f::R3)(x, y, z, r)::Float64 = f.c1*(1-(f.c2*r)+(f.c3*r^2))*exp(-f.α*r/3)

#n = 4
struct R4 <: Radial
    α :: Float64
    n :: Int64
end
R4(α::Float64) = R4(α, 4)
function (f::R4)(x::T, y::T, z::T) where {T<:Float64}
    r = √(x^2 + y^2 + z^2); α = f.α
    Rᵣ = 1/96 * α^(3/2)*(24 - 18*α*r + 3*(α*r)^2 - (α^3*r^3)/8) * exp(-α*r/4)
    return Rᵣ
end

# n = 5
struct R5 <: Radial
    α :: Float64
    n :: Int64
end
R5(α::Float64) = R5(α, 5)
function (f::R5)(x::T, y::T, z::T) where {T<:Float64}
    r = √(x^2 + y^2 + z^2); α = f.α; ρ = 2/5 * α*r
    Rᵣ = 1/(300*√5)*α^(3/2)*(120 - 240*ρ + 120*ρ^2 - 20*ρ^3 + ρ^4)*exp(-ρ/2)
    return Rᵣ
end

# n = 6
struct R6 <: Radial
    α :: Float64
    n :: Int64
end
R6(α::Float64) = R6(α, 6)
function (f::R6)(x::T, y::T, z::T) where {T<:Float64}
    r = √(x^2 + y^2 + z^2); α = f.α; ρ = 1/3 * α*r
    Rᵣ = 1/(2160*√6)*α^(3/2)*(720 - 1800*ρ + 1200*ρ^2 - 300*ρ^3 + 30*ρ^4 - ρ^5)*exp(-ρ/2)
    return Rᵣ
end
"""
    Angular

Angular parts of real atomic orbitals.
"""
abstract type Angular<:OrbitalFunction end

# l = 0
const cs = 1/√(4π)
struct s<:Angular
    l :: Int64
    m :: Int64
end
s() = s(0, 0)

(::s)(x, y, z) = cs
(::s)(x, y, z, r)::Float64 = cs

# l = 1
const cₚ = √(3/4π)
struct px<:Angular
    l :: Int64
    m :: Int64
end
px() = px(1, 1)

function(::px)(x::T, y::T, z::T) where {T<:Float64}
    ϕ = cₚ * x / √(x^2 + y^2 + z^2)
    return ϕ
end
(::px)(x, y, z, r)::Float64 = cₚ * x/r


struct py<:Angular
    l :: Int64
    m :: Int64
end
py() = py(1, -1)

function(::py)(x::T, y::T, z::T) where {T<:Float64}
    ϕ = cₚ * y / √(x^2 + y^2 + z^2)
    return ϕ
end
(::py)(x, y, z, r)::Float64 = cₚ * y/r

struct pz<:Angular
    l :: Int64
    m :: Int64
end
pz() = pz(1, 0)

function(::pz)(x::T, y::T, z::T) where {T<:Float64}
    ϕ = cₚ * z / √(x^2 + y^2 + z^2)
    return ϕ
end
(::pz)(x, y, z, r)::Float64 = cₚ * z/r

# l = 2
const cd1 = √(15/4π)
const cd2 = √(5/16π)
const cd3 = √(15/16π)

struct dxy<:Angular
    l :: Int64
    m :: Int64
end
dxy() = dxy(2, -2)

function(::dxy)(x::T, y::T, z::T) where {T<:Float64}
    ϕ = cd1 * x*y/(x^2 + y^2 + z^2)
    return ϕ
end
(::dxy)(x, y, z, r) = cd1 * x*y/(r^2)


struct dyz<:Angular
    l :: Int64
    m :: Int64
end
dyz() = dyz(2, -1)

function(::dyz)(x::T, y::T, z::T) where {T<:Float64}
    ϕ = cd1 * y*z/(x^2 + y^2 + z^2)
    return ϕ
end
(::dyz)(x, y, z, r) = cd1 * y*z/(r^2)

struct dxz<:Angular
    l :: Int64
    m :: Int64
end
dxz() = dxz(2, 1)

function(::dxz)(x::T, y::T, z::T) where {T<:Float64}
    ϕ = cd1 * x*z/(x^2 + y^2 + z^2)
    return ϕ
end
(::dxz)(x, y, z, r) = cd1 * x*z/(r^2)

struct dz2<:Angular
    l :: Int64
    m :: Int64
end
dz2() = dz2(2, 0)

function(::dz2)(x::T, y::T, z::T) where {T<:Float64}
    r² = (x^2 + y^2 + z^2)
    ϕ = cd2 * (3*z^2 - r²) / r²
    return ϕ
end
(::dz2)(x, y, z, r) = cd2 * (3*z^2 - r^2) / (r^2)

struct dx2_y2<:Angular
    l :: Int64
    m :: Int64
end
dx2_y2() = dx2_y2(2, 2)
function(::dx2_y2)(x::T, y::T, z::T) where {T<:Float64}
    ϕ = cd3 * (x^2 - y^2) / (x^2 + y^2 + z^2)
    return ϕ
end
(::dx2_y2)(x, y, z, r) = cd3 * (x^2 - y^2) / (r^2)

"""
    Overlap(Ψ₁, r⃗₁, Ψ₂, r⃗₂)

Struct to calculate the overlap integral between two wavefunctions with centers
`r⃗₁` and `r⃗₂`.
"""
struct Overlap{F1,F2,A1,A2}<:Function
    Ψ₁ :: F1
    r⃗₁ :: A1
    Ψ₂ :: F2
    r⃗₂ :: A2
end
(ov::Overlap)(r⃗::AbstractArray{Float64, 1})::Float64 = ov.Ψ₁(map(-, r⃗, ov.r⃗₁)) * ov.Ψ₂(map(-, r⃗, ov.r⃗₂))

Overlap(wf1, wf2) = Overlap(wf1, [0., 0., 0.], wf2, [0., 0., 0.])

function Overlap(ϕ₁::A1, R₁::R1, r⃗₁::V1, ϕ₂::A2, R₂::R2, r⃗₂::V2) where {A1,A2<:Angular,R1,R2<:Radial,V1,V2<:AbstractArray{Float64, 1}}
    return f(r⃗) = ϕ₁(r⃗, r⃗₁)*R₁(r⃗, r⃗₁) * ϕ₂(r⃗, r⃗₂) * R₂(r⃗, r⃗₂)
end

struct porb <: Angular
    l :: Int64
end
porb() = porb(1)

struct dorb <: Angular
    l :: Int64
end
dorb() = dorb(2)

struct sp3 <: Angular
    l :: Int64
end
sp3() = sp3(-1)

struct sp3dr2 <: Angular
    l :: Int64
end
sp3dr2() = sp3dr2(-2)

struct prdr2 <: Angular
    l :: Int64
end
prdr2() = prdr2(-3)

struct pxdx2 <: Angular
    l :: Int64
end
pxdx2() = pxdx2(-3)
struct pydy2 <: Angular
    l :: Int64
end
pydy2() = pydy2(-3)
struct pzdz2 <: Angular
    l :: Int64
end
pzdz2() = pzdz2(-3)