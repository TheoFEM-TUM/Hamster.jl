"""
    get_transformed_system(r⃗₁, r⃗₂, ω)

Construct a coordinate system with z-axis oriented along the connecting vector
of `r⃗₁` and `r⃗₂`. A first guess for the x-axis is `ω`.
"""
function get_transformed_system(r⃗₁, r⃗₂, ω; ϵ=1e-5)
    if (1 - ϵ < abs(normalize(r⃗₂ .- r⃗₁) ⋅ normalize(ω)) < 1 + ϵ) == false
        Ê = zeros(Float64, (3, 3))
        ez = ifelse(normdiff(r⃗₂, r⃗₁) > ϵ, normalize(r⃗₂ .- r⃗₁), [0, 0, 1])
        #@show ez
        ex = ω .- proj(ez, ω); normalize!(ex)
        ey = ez × ex; normalize!(ez)
        Ê[1, :] = ex; Ê[2, :] = ey; Ê[3, :] = ez
        return Ê
    else
        return get_transformed_system(r⃗₁, r⃗₂)
    end
end

"""
    get_transformed_system(r⃗₁, r⃗₂)

Construct a coordinate system with z-axis oriented along the connecting vector
of `r⃗₁` and `r⃗₂` by rotating the initial cartesian coordinate system.
"""
function get_transformed_system(r⃗₁, r⃗₂)
    ez = Float64[0, 0, 1]
    ϵ=0.0001; 
    z_new = normdiff(r⃗₁, r⃗₂) > ϵ ? normalize(r⃗₂ .- r⃗₁) : ez
    θ = calc_angle(z_new, ez)
    u⃗ = cross(ez, z_new)
    if norm(u⃗) > ϵ; normalize!(u⃗); end
    U = rotation_matrix_around_axis(u⃗, θ)
    return transpose(U)
end

"""
    rotation_matrix_around_axis(u⃗, θ)

Calculate the rotation matrix for rotations by an angle `θ` around the axis `u⃗`.
"""
function rotation_matrix_around_axis(u⃗, θ)
    R = @. sin(θ) * $cross_prod_matrix(u⃗) + (1-cos(θ)) * $*(u⃗, u⃗') + cos(θ)*Id3
end

const Id3 = Matrix{Float64}(I, 3, 3)
function cross_prod_matrix(v)
    vx = similar(Id3)
    for i in 1:3
        ei = @view Id3[:, i]
        vx[:, i] = v × ei
    end
    return vx
end