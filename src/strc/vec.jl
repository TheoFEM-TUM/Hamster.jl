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
    normdiff(v::AbstractVector, w::AbstractVector)

Compute the Euclidean norm (L2 distance) between two vectors `v` and `w`.

# Arguments
- `v::AbstractVector`: The first vector.
- `w::AbstractVector`: The second vector.

# Returns
- `Float64`: The Euclidean norm of the difference between vectors `v` and `w`.
"""
function normdiff(v⃗::V, w⃗::W) where {V,W}
    out = 0.
    @inbounds @simd for i in eachindex(v⃗)
        @views out += (v⃗[i] - w⃗[i])^2
    end
    return √out
end

function normdiff(v⃗::V, w⃗::W, t⃗::T) where {V,W,T}
    out = 0.
    @views for (vi, wi, ti) in zip(v⃗, w⃗, t⃗)
        out += (vi - (wi - ti))^2
    end
    return √out
end

function normdiff(v⃗, w⃗, δv⃗, δw⃗, t⃗)
    out = 0.
    @views for (vi, wi, δvi, δwi, ti) in zip(v⃗, w⃗, δv⃗, δw⃗, t⃗)
        out += (vi - δvi - (wi - δwi - ti))^2
    end
    return √out
end