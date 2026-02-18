"""
    Orbital{O<:Angular}(ion_type::Int64, type::O, axis::SVector{3, Float64})

Defines an orbital associated with an ion, characterized by its type, angular momentum properties, and orientation.

# Type Parameters
- `O<:Angular`: Specifies the angular momentum type, such as s, p, or d orbitals.

# Fields
- `ion_type::UInt8`: An integer representing the type or species of the ion this orbital is associated with.
- `type::O`: The angular momentum type of the orbital, parameterized by `O`, which defines the orbital's angular properties (e.g., s, p, d orbitals).
- `axis::SVector{3, Float64}`: A 3D unit vector representing the orientation or axis of the orbital in Cartesian coordinates.
"""
struct Orbital{O<:Angular}
    ion_type :: UInt8
    type :: O
    axis :: SVector{3, Float64}
end

"""
    get_orbitals(strc::Structure, conf=get_empty_config())

Generate a list of orbitals for the ions in a given `Structure` object, using a configuration for the orbital types and axes.

# Arguments
- `strc::Structure`: The structure containing ions and their corresponding positions and types.
- `conf`: Configuration object that defines how to retrieve orbital information for the given ions. Defaults to `get_empty_config()` if no configuration is provided.

# Returns
- `orbitals::Vector{Orbital}`: A vector of `Orbital` objects, where each orbital corresponds to a specific ion in the structure.

# Details
- The function loops over all ions in the given structure (`strc`), retrieves the orbital types for each ion based on the ion's type, and generates corresponding axes.
- For each ion, it constructs the associated orbitals, considering the ion's type, angular properties, and orientation (axes).
- It then stores the orbitals as `Orbital` objects in a vector.
"""
function get_orbitals(strc::Structure, conf=get_empty_config())
    orbitals = Vector{Orbital}[]

    Norbs = Int64[]

    for iion in eachindex(strc.ions)
        ion_orbitals = Orbital[]
        orbital_list = str_to_orb.(get_orbitals(conf, number_to_element(strc.ions[iion].type)))
        push!(Norbs, length(orbital_list))
        axes = get_axes(iion, strc, orbital_list, conf)
        for jorb in eachindex(orbital_list)
            push!(ion_orbitals, Orbital(strc.ions[iion].type, orbital_list[jorb], normalize(axes[jorb])))
        end
        push!(orbitals, ion_orbitals)
    end

    return orbitals
end

get_number_of_orbitals(orbitals) = length.(orbitals)

"""
    get_axes_from_orbitals(orbitals::Vector{Vector{Orbitals}}]) -> Vector{Matrix{3, Norb}}

Reads the orbital axes from a nested Vector of `Orbital` and returns them as a Vector of matrices.

# Arguments
- `orbitals::Vector{Vector{Orbital}}`: A vector where each element is a vector of `Orbital`.
"""
get_axes_from_orbitals(orbitals::Vector{Vector{Orbital}}) = [hcat([orb.axis for orb in orb_list]...) for orb_list in orbitals]

"""
    get_axes(iion::Int64, strc::Structure, orbital_list::Vector{Angular}, conf=get_empty_config(); NNaxes=get_nnaxes(conf, type))

Compute the axes (or orientations) for a set of orbitals associated with a specific ion in a structure.

# Arguments
- `iion::Int64`: The index of the ion for which to compute the axes.
- `strc::Structure`: A `Structure` object representing the lattice and ion configuration.
- `orbital_list::Vector{Angular}`: A vector containing angular orbital types (e.g., `s`, `p`, `sp3`) for which axes need to be computed.
- `conf`: Configuration object (optional) that contains settings for computing axes. Default is `get_empty_config()`.
- `NNaxes`: A Boolean flag that determines whether to compute nearest-neighbor axes (`true`) or directly use predefined orbital axes (`false`).
"""
function get_axes(iion::Int64, strc::Structure, orbital_list, conf=get_empty_config(); NNaxes=get_nnaxes(conf, number_to_element(strc.ions[iion].type)))
    axes = SVector{3, Float64}[]
    if NNaxes
        rs_ion = get_ion_positions(strc.ions)
        rNNs = get_nearest_neighbors(strc.ions[iion].pos, rs_ion, frac_to_cart(strc.Rs, strc.lattice), strc.point_grid, kNN=length(orbital_list))
        return sort([rNN - strc.ions[iion].pos for rNN in rNNs], rev=true)
    else
        if all(h->typeof(h)==sp3, orbital_list) || all(h->typeof(h)==sp3dr2, orbital_list)
            return get_axis(sp3())
        else
            for orbital in orbital_list
                push!(axes, get_axis(orbital))
            end
            return axes
        end
    end
end

"""
    get_axis(h::HybridOrbital) -> Vector{SVector{3, Float64}}
    get_axis(o::AngularOrbital) -> SVector{3, Float64}

Returns the unit vector(s) representing the axis (or axes) of the given orbital type.

# Orbital Types and Axes
- `h::sp3`: Returns four unit vectors corresponding to the axes of `sp3` (or `sp3dr2`) hybrid orbitals: `[1, 1, 1]`, `[1, -1, -1]`, `[-1, 1, -1]`, `[-1, -1, 1]`.
- `o::s`: Returns the axis `[0, 0, 1]` for an `s` orbital.
- `o::px`: Returns the axis `[1, 0, 0]` for a `px` orbital.
- `o::py`: Returns the axis `[0, 1, 0]` for a `py` orbital.
- `o::pz`: Returns the axis `[0, 0, 1]` for a `pz` orbital.
- `o::pxdx2`: Returns the axis `[1, 0, 0]` for a combined `px/dx²` orbital.
- `o::pydy2`: Returns the axis `[0, 1, 0]` for a combined `py/dy²` orbital.
- `o::pzdz2`: Returns the axis `[0, 0, 1]` for a combined `pz/dz²` orbital.
"""
get_axis(h::sp3) = [SVector{3}([1., 1., 1.]), 
                    SVector{3}([1., -1., -1.]), 
                    SVector{3}([-1., 1., -1.]), 
                    SVector{3}([-1., -1., 1.])]
get_axis(o) = SVector{3}([0., 0., 1.])

function get_sym_axis(orb::Orbital) :: SVector{3, Float64}
    if orb.type isa sp3 || orb.type isa sp3dr2
        return orb.axis
    else
        return get_sym_axis(orb.type)
    end
end
get_sym_axis(o::s) = SVector{3}([0., 0., 1.])
get_sym_axis(o::px) = SVector{3}([1., 0., 0.]); get_sym_axis(o::py) = SVector{3}([0., 1., 0.]); get_sym_axis(o::pz) = SVector{3}([0., 0., 1.])
get_sym_axis(o::dxz) = SVector{3}([1., 0., 1.]); get_sym_axis(o::dxy) = SVector{3}([1., 1., 0.]); 
get_sym_axis(o::dyz) = SVector{3}([0., 1., 1.]); get_sym_axis(o::dz2) = SVector{3}([0., 0., 1.]); 
get_sym_axis(o::dx2_y2) = SVector{3}([1., -1., 0.])


get_orbital_list(::s) = Angular[s()]
get_orbital_list(p::px) = Angular[px(), py(), py()]
get_orbital_list(p::py) = Angular[px(), py(), pz()]
get_orbital_list(p::pz) = Angular[px(), py(), pz()]
get_orbital_list(h::sp3) = Angular[s(), px(), py(), pz()]
get_orbital_list(d::dxz) = [dxz(), dxy(), dyz(), dx2_y2(), dz2()]
get_orbital_list(d::dxy) = [dxz(), dxy(), dyz(), dx2_y2(), dz2()]
get_orbital_list(d::dyz) = [dxz(), dxy(), dyz(), dx2_y2(), dz2()]
get_orbital_list(d::dz2) = [dxz(), dxy(), dyz(), dx2_y2(), dz2()]
get_orbital_list(d::dx2_y2) = [dxz(), dxy(), dyz(), dx2_y2(), dz2()]
get_orbital_list(h::sp3dr2) = Angular[s(), px(), py(), pz(), dxz(), dxy(), dyz(), dx2_y2(), dz2()]