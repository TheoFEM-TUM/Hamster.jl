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

Create a `Structure` object by reading a POSCAR file, which contains information about the crystal structure, including lattice vectors, atomic positions, and atom types.

# Arguments
- `conf`: A configuration object that contains various parameters like file paths, cutoff radius, and grid size. Defaults to an empty configuration.
- `poscar_path`: The file path to the POSCAR file, which is read to obtain the crystal structure information. Derived from `conf` if not provided.
- `rcut`: The cutoff radius for determining which translation vectors to include based on interatomic distances. Derived from `conf` if not provided.
- `grid_size`: The size of the grid used in the `PointGrid` for efficient neighbor searching. Derived from `conf` if not provided.

# Returns
- `Structure`: A `Structure` object that includes the lattice vectors, translation vectors, ions, and a point grid, initialized based on the POSCAR file.
"""
function Structure(conf=get_empty_config(); poscar_path=get_poscar(conf), rcut=get_rcut(conf), grid_size=get_grid_size(conf))
    poscar = read_poscar(poscar_path)

    return Structure(frac_to_cart(poscar.Rs, poscar.lattice), poscar.atom_types, poscar.lattice, conf, rcut=rcut, grid_size=grid_size)
end

"""
    Structure(rs_ion, δrs_ion=zeros(3, size(rs_ion, 2)), ion_types, lattice, conf=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf))

Create a `Structure` object representing a crystal structure, including its lattice vectors, atomic positions, and a point grid for neighbor searching.

# Arguments
- `rs_ion`: A 3xN matrix representing the fractional coordinates of the ions in the unit cell, where N is the number of ions.
- `δrs_ion`: A 3xN matrix representing small displacements from the positions in `rs_ion`. Defaults to a zero matrix if no displacements are provided.
- `ion_types`: A vector representing the types of ions in the unit cell.
- `lattice`: A 3x3 matrix representing the lattice vectors of the crystal. Each column corresponds to a lattice vector.
- `conf`: A configuration object that may contain parameters like cutoff radius and grid size. Defaults to an empty configuration.
- `rcut`: The cutoff radius for determining which translation vectors to include based on interatomic distances. Derived from `conf` if not provided.
- `grid_size`: The size of the grid used in the `PointGrid` for efficient neighbor searching. Derived from `conf` if not provided.

# Returns
- `Structure`: A `Structure` object that includes the lattice vectors, translation vectors, ions, and a point grid.
"""
function Structure(rs_ion::Matrix{Float64}, ion_types, lattice::Matrix{Float64}, δrs_ion=zeros(3, size(rs_ion, 2)), conf=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf))
    Rs = get_translation_vectors(rs_ion, lattice, rcut=rcut)

    point_grid = PointGrid(rs_ion, frac_to_cart(Rs, lattice), grid_size=grid_size)
    ions = get_ions(rs_ion, ion_types, δrs_ion)

    return Structure(lattice, Rs, ions, point_grid)
end