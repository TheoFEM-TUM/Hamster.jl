const ê_z = Float64[0, 0, 1]

"""
    get_sk_transform_matrix(r⃗₁, r⃗₂, axis, tmethod)

Returns a transformation matrix that aligns the z-axis along the vector connecting 
`r⃗₁` and `r⃗₂`. The method of transformation depends on `tmethod`.

# Arguments
- `r⃗₁::AbstractVector`: The first position vector.
- `r⃗₂::AbstractVector`: The second position vector.
- `axis::AbstractVector`: A vector used to define the x-axis in case of Gram-Schmidt transformation.
- `tmethod::String`: The transformation method. 

# Returns
- `Ê::AbstractMatrix`: The 3x3 transformation matrix that aligns the z-axis with the vector 
  between `r⃗₁` and `r⃗₂`.
"""
function get_sk_transform_matrix(r⃗₁, r⃗₂, axis, tmethod)
  if lowercase(tmethod)[1] == 'g'
      return get_transformed_system(r⃗₁, r⃗₂, axis)
  else
      return get_transformed_system(r⃗₁, r⃗₂)
  end
end

"""
    get_transformed_system(r⃗₁, r⃗₂, ω; ϵ=1e-5)
    get_transformed_system(r⃗₁, r⃗₂)

Construct a new orthonormal coordinate system based on two vectors `r⃗₁` and `r⃗₂`, 
with the z-axis oriented along the vector connecting `r⃗₁` and `r⃗₂`. Optionally, a 
third vector `ω` can be provided to influence the initial guess for the x-axis.

# Arguments
- `r⃗₁::Vector`: A 3D vector representing the first position.
- `r⃗₂::Vector`: A 3D vector representing the second position.
- `ω::Vector`: (Optional) A 3D vector used as the initial guess for the x-axis orientation. 
  If `ω` is omitted, the function assumes a default method for constructing the x-axis.
- `ϵ::Float64=1e-5`: (Optional) A small tolerance value to handle near-zero differences and near-parallel vectors.

# Returns
- A 3x3 matrix `Ê` where the rows correspond to the new x-axis, y-axis, and z-axis of the coordinate system.

# Details
- **Z-axis (ez)**: Always aligned with the normalized vector `r⃗₂ - r⃗₁`. If this vector is zero or near-zero 
  (within tolerance `ϵ`), the z-axis defaults to `[0, 0, 1]`.
- **X-axis (ex)**: If `ω` is provided, the x-axis is constructed from the projection of `ω` orthogonal to the 
  z-axis and is normalized. If `ω` is omitted, an alternative method will define the x-axis.
- **Y-axis (ey)**: Always computed as the cross-product of the z-axis and x-axis, ensuring orthogonality and 
  completeness of the coordinate system.
- **Fallback behavior**: If `ω` is nearly parallel to the z-axis within tolerance `ϵ`, the function reverts to 
  the simpler version `get_transformed_system(r⃗₁, r⃗₂)` to handle the ambiguity in the x-axis.
"""
function get_transformed_system(r⃗₁, r⃗₂, ω; ϵ=1e-5)
    if (1 - ϵ < abs(normalize(r⃗₂ .- r⃗₁) ⋅ normalize(ω)) < 1 + ϵ) == false
        Ê = zeros(Float64, (3, 3))
        ez = ifelse(normdiff(r⃗₂, r⃗₁) > ϵ, normalize(r⃗₂ .- r⃗₁), ê_z)
        ex = ω .- proj(ez, ω); normalize!(ex)
        ey = ez × ex; normalize!(ez)
        Ê[1, :] = ex; Ê[2, :] = ey; Ê[3, :] = ez
        return SMatrix{3, 3}(Ê)
    else
        return get_transformed_system(r⃗₁, r⃗₂)
    end
end

function get_transformed_system(r⃗₁, r⃗₂; ϵ=1e-5)
    z_new = normdiff(r⃗₁, r⃗₂) > ϵ ? normalize(r⃗₂ .- r⃗₁) : ê_z
    θ = calc_angle(z_new, ê_z)
    u⃗ = cross(ê_z, z_new)
    if norm(u⃗) > ϵ; normalize!(u⃗); end
    U = rotation_matrix_around_axis(u⃗, θ)
    return transpose(U)
end

"""
    rotation_matrix_around_axis(u⃗::AbstractVector{T}, θ::T) where T

Constructs a 3x3 rotation matrix that rotates by an angle `θ` around an arbitrary axis `u⃗`, using Rodrigues' rotation formula.

# Arguments
- `u⃗::AbstractVector{T}`: A 3D unit vector (axis of rotation) where `T` is a numeric type.
- `θ::T`: The angle of rotation (in radians) around the axis `u⃗`.

# Returns
- A 3x3 static matrix representing the rotation around the axis `u⃗` by angle `θ`.
"""
function rotation_matrix_around_axis(u⃗::AbstractVector{T}, θ::T) where T
    cosθ = cos(θ)
    sinθ = sin(θ)
    ux, uy, uz = u⃗

    # Construct the rotation matrix using Rodrigues' formula
    U = @SMatrix [
        cosθ + ux^2*(1 - cosθ)    ux*uy * (1 - cosθ) - uz*sinθ  ux*uz * (1 - cosθ) + uy*sinθ;
        uy*ux * (1 - cosθ) + uz * sinθ  cosθ + uy^2 * (1 - cosθ)    uy*uz * (1 - cosθ) - ux*sinθ;
        uz*ux * (1 - cosθ) - uy*sinθ  uz*uy * (1 - cosθ) + ux*sinθ  cosθ + uz^2 * (1 - cosθ)
    ]
    
    return U
end