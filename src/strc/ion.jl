"""
    Ion

A mutable structure representing an ion in a crystal lattice.

# Fields
- `type::UInt8`: The type or species of the ion, denoted by its proton number.
- `pos::StaticArray{3, Float64}`: A 3D static array representing the position of the ion in Cartesian coordinates.
- `dist::StaticArray{3, Float64}`: A 3D static array representing any distortion applied to the ion's position.
"""
mutable struct Ion
    type :: UInt8
    pos :: SVector{3, Float64}
    dist :: SVector{3, Float64}
end

"""
    get_ions(positions, types, distortions=zeros(3, size(positions, 2)))

Create a vector of `Ion` instances from given positions, types, and distortions.

# Arguments
- `positions::AbstractMatrix{T}`: A matrix where each column represents the 3D position of an ion in Cartesian coordinates.
- `types::AbstractVector{String}`: A vector of strings representing the type or species of each ion, corresponding to the columns of `positions`.
- `distortions::AbstractMatrix{T}`: A matrix where each column represents the 3D distortion vector applied to the corresponding ion's position. Defaults to a matrix of zeros.

# Returns
- `Vector{Ion}`: A vector of `Ion` instances, each containing the type, position, and distortion of an ion.
"""
function get_ions(positions, types, distortions=zeros(3, size(positions, 2)))
    ions = Ion[]
    for iion in axes(positions, 2)
        push!(ions, Ion(types[iion], SVector{3}(positions[:, iion]), SVector{3}(distortions[:, iion])))
    end
    return ions
end

"""
    get_ion_types(ions::Vector{Ion}; uniq=false, sorted=false)

Return an array containing the types of all ions in the vector `ions`.

# Arguments
- `ions::Vector{Ion}`: A vector of `Ion` instances, each containing information about an ion's type, position, and distortion.
- `uniq::Bool=false`: If `true`, the returned array contains only unique ion types.
- `sorted::Bool=false`: If `true`, the returned array is sorted in alphabetical order.

# Returns
- `Vector{String}`: An array of ion types. The array will contain all ion types present in the input vector `ions`. If `uniq` is set to `true`, only unique types will be included. If `sorted` is set to `true`, the types will be sorted alphabetically.
"""
function get_ion_types(ions::Vector{Ion}; uniq=false, sorted=false)
    ion_types = Vector{UInt8}(undef, length(ions))
    for (iion, ion) in enumerate(ions)
        ion_types[iion] = ion.type
    end
    if uniq; ion_types = unique(ion_types); end
    if sorted; sort!(ion_types); end
    return ion_types
end

"""
    findnext_ion_of_type(type, ions::Vector{Ion}) -> Int64

Find the index of the next ion in the vector `Ions` that has the specified `type`.

# Arguments:
- `type`: The type of ion to search for. This could be a string, integer, or any other type that represents an ion type.
- `Ions`: A vector of `Ion` objects, where each `Ion` has a `type` field that specifies its ion type.

# Returns:
- The index `iion` of the first ion in `ions` whose `type` matches the input `type`. 
- If no ion with the specified `type` is found, the function returns `0`.
"""
function findnext_ion_of_type(ion_type::Integer, ions::Vector{Ion})
    for iion in eachindex(ions)
        if ions[iion].type == ion_type; return iion; end
    end
    return 0
end

function findnext_ion_of_type(ion_type, ions::Vector{Ion})
    type = element_to_number(ion_type)
    for iion in eachindex(ions)
        if ions[iion].type == type; return iion; end
    end
    return 0
end

"""
    get_ion_positions(ions::Vector{Ion}) :: Vector{SVector{3, Float64}}

Retrieve the positions of all ions in a given vector of `Ion` objects.

# Arguments
- `ions::Vector{Ion}`: A vector of `Ion` objects, where each `Ion` contains information about its type, position, and distortion.

# Returns
- `Vector{SVector{3, Float64}}`: A vector of `SVector{3, Float64}` where each element represents the 3D Cartesian coordinates of an ion.
"""
get_ion_positions(ions::Vector{Ion}; apply_distortion=false)::Vector{SVector{3, Float64}} = apply_distortion ? [ion.pos - ion.dist for ion in ions] : [ion.pos for ion in ions]