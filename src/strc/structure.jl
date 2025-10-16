"""
    Structure

A data structure representing a crystal structure, including its lattice vectors, atomic positions, and a point grid for efficient neighbor searching.

# Fields
- `lattice::Matrix{Float64}`: A 3x3 matrix representing the lattice vectors of the crystal. Each column corresponds to a lattice vector.
- `Rs::Matrix{Float64}`: A matrix where each column represents the position of a lattice translation vector in fractional coordinates.
- `ions::Vector{Ion}`: A vector containing the ions in the structure. Each `Ion` includes information like the type of ion, its position, and possibly its distortion from the equilibrium.
- `point_grid::PointGrid`: A data structure used for efficiently finding neighboring ions based on their positions.
"""
struct Structure
    lattice :: Matrix{Float64}
    Rs :: Matrix{Float64}
    ions :: Vector{Ion}
    point_grid :: PointGrid
end

"""
    Structure(conf=get_empty_config(); poscar_path=get_poscar(conf), rcut=get_rcut(conf), grid_size=get_grid_size(conf))

Create a `Structure` instance from a POSCAR file.

# Arguments
- `conf`: A `Config` instance that contains various parameters.
- `poscar_path`: The file path to the POSCAR file.
- `rcut`: The cutoff radius for interactions to be taken into account.
- `grid_size`: The size of the grid used in the `PointGrid` for efficient neighbor searching.

# Returns
- `Structure`: A `Structure` instance.
"""
function Structure(conf=get_empty_config(); Rs=zeros(3, 1), poscar_path=get_poscar(conf), rcut=get_rcut(conf), grid_size=get_grid_size(conf), verbosity=get_verbosity(conf))
    time = @elapsed begin
        poscar = read_poscar(poscar_path)
        @unpack rs_atom, lattice, atom_types = poscar
        Rs = Rs == zeros(3, 1) ? get_translation_vectors(frac_to_cart(rs_atom, lattice), lattice, rcut=rcut) : Rs
        strc = Structure(Rs, frac_to_cart(rs_atom, lattice), atom_types, lattice, conf, 
                            rcut=rcut, grid_size=grid_size)
    end
    if verbosity > 0
        append_output_block("Structure Information:", 
        ["POSCAR", "rcut", "grid_size", "Number of atoms", "Unique atom species", "Number of R vectors", "Ion interactions", "Ion interaction total", "Structure time"], 
        [poscar_path, rcut, grid_size, length(strc.ions), get_ion_types(strc.ions, uniq=true), size(strc.Rs, 2), length(iterate_nn_grid_points(strc.point_grid)), length(strc.ions)^2 * size(strc.Rs, 2), time])
    end
    return strc
end

"""
    Structure(Rs, rs_ion, δrs_ion, ion_types, lattice, conf=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf))
    Structure(Rs, rs_ion, ion_types, lattice, conf=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf))

Create a `Structure` instance from atomic information.

# Arguments
- `Rs`: A 3xNR matrix representing the set of lattice translation vectors.
- `rs_ion`: A 3xNion matrix representing the coordinates of the ions in the unit cell.
- `δrs_ion`: (Optional) A 3xNion matrix representing atomic displacements from `rs_ion`. Defaults to a zeros.
- `ion_types`: A vector representing the types of ions in the unit cell.
- `lattice`: A 3x3 matrix representing the lattice vectors of the crystal.
- `conf`: (Optional) A `Config` instance.
- `rcut`: The cutoff radius for interactions to be taken into account.
- `grid_size`: The size of the grid used in the `PointGrid` for efficient neighbor searching.

# Returns
- `Structure`: A `Structure` instance.
"""
function Structure(Rs::M1, rs_ion::M2, δrs_ion::M3, ion_types, lattice::M4, conf::Config=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf)) where {M1,M2,M3,M4<:Matrix{Float64}}
    point_grid = PointGrid(rs_ion, frac_to_cart(Rs, lattice), grid_size=grid_size)
    ions = get_ions(rs_ion, ion_types, δrs_ion)

    return Structure(lattice, Rs, ions, point_grid)
end

function Structure(Rs::M1, rs_ion::M2, ion_types, lattice::M3, conf::Config=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf)) where {M1,M2,M3<:Matrix{Float64}}
    return Structure(Rs, rs_ion, zeros(3, size(rs_ion, 2)), ion_types, lattice, conf; rcut=rcut, grid_size=grid_size)
end