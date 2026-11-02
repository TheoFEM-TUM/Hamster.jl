const ê_z = Float64[0, 0, 1]

"""
    get_sk_transform_matrix(r⃗₁, r⃗₂; ϵ=1e-5)

Rotation matrix whose rows are the axes of the bond frame of the bond from `r⃗₁` to `r⃗₂`. The z-axis points along
`r⃗₂ - r⃗₁` (along `[0, 0, 1]` if the positions coincide within `ϵ`); the frame is the global one rotated around
`ê_z × (r⃗₂ - r⃗₁)`.
"""
function get_sk_transform_matrix(r⃗₁, r⃗₂; ϵ=1e-5)
    z_new = normdiff(r⃗₁, r⃗₂) > ϵ ? normalize(r⃗₂ .- r⃗₁) : ê_z
    θ = calc_angle(z_new, ê_z)
    u⃗ = cross(ê_z, z_new)
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