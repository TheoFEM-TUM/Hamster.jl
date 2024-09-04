"""
    Ion

A mutable structure representing an ion in a crystal lattice.

# Fields
- `type::String`: The type or species of the ion, usually denoted by its chemical symbol (e.g., "Na" for sodium, "Cl" for chlorine).
- `pos::StaticArray{3, Float64}`: A 3D static array representing the position of the ion in Cartesian coordinates.
- `dist::StaticArray{3, Float64}`: A 3D static array representing any distortion applied to the ion's position.
"""
mutable struct Ion
    type :: String
    pos :: SVector{3, Float64}
    dist :: SVector{3, Float64}
end

"""
    get_ions(positions, types, distortions)

Create a vector of `Ion` instances from given positions, types, and distortions.

# Arguments
- `positions::AbstractMatrix{T}`: A matrix where each column represents the 3D position of an ion in Cartesian coordinates.
- `types::AbstractVector{String}`: A vector of strings representing the type or species of each ion, corresponding to the columns of `positions`.
- `distortions::AbstractMatrix{T}`: A matrix where each column represents the 3D distortion vector applied to the corresponding ion's position.

# Returns
- `Vector{Ion}`: A vector of `Ion` instances, each containing the type, position, and distortion of an ion.
"""
function get_ions(positions, types, distortions)
    ions = Ion[]
    for iion in axes(rs_ion, 2)
        push!(ions, Ion(types[iion], SVector{3}(positions[:, iion]), SVector{3}(distortions[:, iion])))
    end
    return ions
end