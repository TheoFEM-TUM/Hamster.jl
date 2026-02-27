"""
    transform_basis(r⃗::AbstractVector, Ê::AbstractMatrix) -> AbstractVector

Transforms a vector to a new basis using a given transformation matrix.

This function takes a vector `r⃗` and transforms it into a new basis by applying the transformation matrix `Ê`. The transformation is performed by multiplying the matrix `Ê` with the vector `r⃗`, yielding the transformed vector.

# Arguments:
- `r⃗::AbstractVector`: The vector to be transformed. This represents coordinates or a state in the original basis.
- `Ê::AbstractMatrix`: The transformation matrix that defines the new basis. This matrix should have dimensions compatible with the vector `r⃗`.

# Returns:
- `AbstractVector`: The transformed vector in the new basis, obtained by multiplying `Ê` with `r⃗`.

"""
transform_basis(r⃗, Ê) = Ê * r⃗

frac_to_cart(r⃗_frac, lattice) = transform_basis(r⃗_frac, lattice)
cart_to_frac(r⃗_cart, lattice) = transform_basis(r⃗_cart, inv(lattice))

"""
    transform_to_spherical(r⃗::AbstractArray{Float64, 1}) -> Tuple{Float64, Float64, Float64}

Converts Cartesian coordinates to spherical coordinates.

This function takes a 3D vector `r⃗` in Cartesian coordinates (x, y, z) and transforms it into spherical coordinates (r, θ, φ). The spherical coordinates are defined as follows:

- `r`: The radial distance from the origin.
- `θ`: The polar angle, measured from the positive z-axis.
- `φ`: The azimuthal angle, measured from the positive x-axis in the xy-plane.

# Arguments:
- `r⃗::AbstractArray{Float64, 1}`: A 3-element vector representing the Cartesian coordinates `[x, y, z]`.

# Returns:
- `Tuple{Float64, Float64, Float64}`: A tuple containing the spherical coordinates `(r, θ, φ)`.

"""
function transform_to_spherical(r⃗::AbstractArray{Float64, 1})
    x, y, z = r⃗
    r = √(x^2 + y^2 + z^2)
    φ = atan(y, x)
    θ = atan(√(x^2 + y^2), z)
    return r, θ, φ
end

"""
    transform_to_spherical(r⃗::AbstractArray{Float64, 2}; origin=[0, 0, 0]) -> AbstractMatrix{Float64}

Converts a set of Cartesian coordinates to spherical coordinates relative to a specified origin.

This function takes a 3xN matrix `r⃗`, where each column represents a point in Cartesian coordinates (x, y, z), and transforms these points into spherical coordinates (r, θ, φ). The transformation is performed relative to a specified origin.

# Arguments:
- `r⃗::AbstractArray{Float64, 2}`: A 3xN matrix where each column represents Cartesian coordinates `[x, y, z]` of a point.
- `origin::AbstractVector{Float64}`: A 3-element vector specifying the origin relative to which the spherical coordinates are computed. Default is `[0, 0, 0]`.

# Returns:
- `AbstractMatrix{Float64}`: An N×3 matrix where each row contains the spherical coordinates `(r, θ, φ)` for the corresponding point in the input.
"""
function transform_to_spherical(r⃗::AbstractArray{Float64, 2}; origin=[0,0,0])
    x = @view r⃗[1, :]; y = @view r⃗[2, :]; z = @view r⃗[3, :]
    x′ = x .- origin[1]
    y′ = y .- origin[2]
    z′ = z .- origin[3]

    r = map(√, x′.^2 .+ y′.^2 .+ z′.^2)
    φ = map(atan, y′, x′)
    xy = map(√, x′.^2 .+ y′.^2)
    θ = map(atan, xy, z′)
    return transpose([r θ φ])
end

"""
    normdiff(v⃗::AbstractVector, w⃗::AbstractVector)
    normdiff(v⃗::AbstractVector, w⃗::AbstractVector, t⃗::AbstractVector)
    normdiff(v⃗::AbstractVector, w⃗::AbstractVector, δv⃗::AbstractVector, δw⃗::AbstractVector, t⃗::AbstractVector)

Compute the Euclidean norm (L2 distance) between two vectors `v⃗` and `w⃗`, optionally with displacement vectors `δv⃗` and `δw⃗` and lattice translation vector `t⃗`.

# Arguments
- `v⃗::AbstractVector`: The first vector.
- `w⃗::AbstractVector`: The second vector.
- `δv⃗::AbstractVector`: The first displacement vector.
- `δw⃗::AbstractVector`: The second displacement vector.
- `t⃗::AbstractVector`: The lattice translation vector.

# Returns
- `Float64`: The Euclidean norm of the difference between vectors `v` and `w`.
"""
function normdiff(v⃗::V, w⃗::W) where {V,W<:AbstractVector}
    out = zero(promote_type(eltype(v⃗), eltype(w⃗))) 
    @inbounds for i in eachindex(v⃗)
        @views out += (v⃗[i] - w⃗[i])^2
    end
    return √out
end

function normdiff(v⃗::V, w⃗::W, σ⃗::Vector{Float64}, gauss_width_flag::Bool) where {V,W <:AbstractVector}
    out = zero(promote_type(eltype(v⃗), eltype(w⃗), eltype(σ⃗))) 
    @inbounds for i in eachindex(v⃗)
        @views out += ((v⃗[i] - w⃗[i])/σ⃗[i])^2 /2
    end
    return √out
end

function normdiff(v⃗::V, w⃗::W, t⃗::T) where {V,W,T<:AbstractVector}
    out = zero(promote_type(eltype(v⃗), eltype(w⃗), eltype(t⃗)))
    @inbounds for (vi, wi, ti) in zip(v⃗, w⃗, t⃗)
        @views out += (vi - (wi - ti))^2
    end
    return √out
end

function normdiff(v⃗::V, w⃗::W, δv⃗::DV, δw⃗::DW, t⃗::T) where {V,W,DV,DW,T<:AbstractVector}
    out = zero(promote_type(eltype(v⃗), eltype(w⃗), eltype(δv⃗), eltype(δw⃗), eltype(t⃗)))
    @inbounds for (vi, wi, δvi, δwi, ti) in zip(v⃗, w⃗, δv⃗, δw⃗, t⃗)
        @views out += (vi - δvi - (wi - δwi - ti))^2
    end
    return √out
end

"""
    proj(u⃗, v⃗)

Calculate the projection of the vector `v⃗` onto the vector `u⃗`.

# Arguments
- `u⃗::AbstractVector`: The vector onto which the projection is calculated.
- `v⃗::AbstractVector`: The vector being projected onto `u⃗`.

# Returns
- `AbstractVector`: The projection of `v⃗` onto `u⃗`.
"""
function proj(u⃗, v⃗)
    if !(norm(u⃗) ≈ 0)
        return (u⃗⋅v⃗)/(u⃗⋅u⃗) .* u⃗
    else
        return zero(u⃗)        
    end
end

"""
    calc_angle(v1, v2; ϵ=1e-5)

Calculate the angle between two vectors `v1` and `v2` in radians. 

# Arguments:
- `v1`: First vector (can be any dimensionality as long as it matches `v2`).
- `v2`: Second vector (same dimensionality as `v1`).
- `ϵ`: Small tolerance value to ensure that the norm of the vectors is sufficiently large to avoid division by zero (default: `1e-5`).

# Returns:
- The angle in radians between `v1` and `v2`, calculated using the dot product. 
  If either vector's norm is less than `ϵ`, the function returns `0.`.
"""
function calc_angle(v1, v2; ϵ=1e-5)
    if norm(v1) > ϵ && norm(v2) > ϵ
        return acos(round(v1 ⋅ v2 / (norm(v1)*norm(v2)), digits=5))
    else
        return 0.
    end
end

"""
    get_rotated_angles(Û, r⃗)

Calculate the spherical angles (θ, φ) for a vector `r⃗` after it is rotated by the matrix `Û`.

# Arguments
- `Û::AbstractMatrix{T}`: A 3x3 rotation matrix that transforms the vector `r⃗`.
- `r⃗::AbstractVector{T}`: A 3D vector to be rotated, where `T` is a numeric type.

# Returns
- `θ::T`: The polar (zenith) angle, measured from the z-axis, in radians.
- `φ::T`: The azimuthal angle, measured from the x-axis in the xy-plane, in radians.
"""
function get_rotated_angles(Û, r⃗)
    x, y, z = Û * r⃗
    φ = atan(y, x)
    θ = atan(√(x^2 + y^2), z)
    return θ, φ
end
