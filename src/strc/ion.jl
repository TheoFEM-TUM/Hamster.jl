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
get_ion_position(ions::Vector{Ion}, iion; apply_distortion=false)::SVector{3, Float64} = apply_distortion ? ions[iion].pos - ions[iion].dist : ions[iion].pos

function proton_to_position(Z::UInt8)
    # Periodic table layout basics:
    # Periods: 1 to 7
    # Groups: 1 to 18

    if Z < 1 || Z > 118
        error("Proton number out of known periodic table range (1-118).")
    end

    # Precompute element counts per period:
    # Period 1: 2 elements (H, He)
    # Period 2 and 3: 8 elements each
    # Period 4 and 5: 18 elements each
    # Period 6 and 7: 32 elements each (including lanthanides and actinides)
    period_limits = [2, 10, 18, 36, 54, 86, 118]

    # Find period
    period = findfirst(x -> Z <= x, period_limits)

    # Calculate position within the period
    prev_limit = period == 1 ? 0 : period_limits[period - 1]
    pos_in_period = Z - prev_limit

    # Map proton number to group (column)
    # Simplified mapping based on known periodic table layout:

    if period == 1
        # Period 1: H(1) group 1, He(2) group 18
        group = (Z == 1) ? 1 : 18
    elseif period == 2 || period == 3
        # Period 2 and 3: 8 elements
        # Groups 1-2: alkali and alkaline earth metals
        # Groups 13-18: p-block
        if pos_in_period <= 2
            group = pos_in_period  # 1 or 2
        else
            group = pos_in_period + 10  # groups 13 to 18 mapped as 3 -> 13 etc.
        end
    elseif period == 4 || period == 5
        # Period 4 and 5: 18 elements
        # Groups 1-2, 3-12 (transition metals), 13-18
        if pos_in_period <= 2
            group = pos_in_period
        elseif pos_in_period <= 12
            group = pos_in_period  # transition metals 3-12
        else
            group = pos_in_period + 10  # p-block 13-18
        end
    elseif period == 6 || period == 7
        # Period 6 and 7: 32 elements
        # Groups 1-2, 3-12 (transition metals), 13-18 (p-block)
        # Lanthanides/actinides treated as group 3 in main table
        if pos_in_period <= 2
            group = pos_in_period
        elseif pos_in_period <= 12
            group = pos_in_period
        elseif pos_in_period >= 13 && pos_in_period <= 30
            # Lanthanides/actinides not in main group columns,
            # normally shown separately, but here assign group 3
            group = 3
        else
            group = pos_in_period - 14 + 13
        end
    else
        error("Period not handled.")
    end

    return (period, group)
end
