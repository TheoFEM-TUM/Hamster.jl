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
    system :: String
    lattice :: Matrix{Float64}
    Rs :: Matrix{Float64}
    ions :: Vector{Ion}
    point_grid :: PointGrid
end

"""
    rcut_search(conf=get_empty_config(), dmax=0.0) -> Float64

Radius that the neighbor search (grid cell size and translation vectors) must cover. With
`rcut_pair_selection = "reference"`, this is `rcut`. With `"frame"`, it is `rcut + max(rcut_tol, 0) + 2 dmax`, where
`dmax` is the maximum displacement of an ion from its reference position: a pair within the cutoff of the
distorted positions has a reference distance of at most `2 dmax` more, and the neighbor grid finds all pairs with
reference distance up to the grid cell size.

# Arguments
- `conf`: A `Config` instance that contains various parameters.
- `dmax`: Maximum norm of the ion distortions (`dist`) of the structures the search is built for.
"""
rcut_search(conf=get_empty_config(), dmax=0.0) =
    get_rcut_pair_selection(conf) == "frame" ? get_rcut(conf) + max(get_rcut_tol(conf), 0.0) + 2 * dmax : get_rcut(conf)

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
function Structure(conf=get_empty_config(); Rs=zeros(3, 1), poscar_path=get_poscar(conf), rcut=get_rcut(conf), grid_size=get_grid_size(conf), system=get_system(conf), verbosity=get_verbosity(conf))
    time = @elapsed begin
        poscar = read_poscar(poscar_path)
        @unpack rs_atom, lattice, atom_types = poscar
        Rs = Rs == zeros(3, 1) ? get_translation_vectors(frac_to_cart(rs_atom, lattice), lattice, rcut=rcut_search(conf)) : Rs
        strc = Structure(Rs, frac_to_cart(rs_atom, lattice), atom_types, lattice, conf, 
                            rcut=rcut, grid_size=grid_size, system=system)
    end
    if verbosity > 0
        append_output_block("Structure Information:", 
        ["POSCAR", "rcut", "grid_size", "Number of atoms", "Number of R vectors", "Ion interactions", "Ion interaction total", "Structure time"], 
        [poscar_path, rcut, strc.point_grid.grid_size, length(strc.ions), size(strc.Rs, 2), count_nn_pairs(strc.point_grid), length(strc.ions)^2 * size(strc.Rs, 2), time])
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
function Structure(Rs::M1, rs_ion::M2, δrs_ion::M3, ion_types, lattice::M4, conf::Config=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf), system=get_system(conf)) where {M1,M2,M3,M4<:Matrix{Float64}}
    check_cutoffs(conf, rcut=rcut, grid_size=grid_size)
    dmax = isempty(δrs_ion) ? 0.0 : maximum(norm, eachcol(δrs_ion))
    r_search = rcut_search(conf, dmax)
    if r_search > grid_size
        @warn "rcut_pair_selection = \"frame\": grid_size = $grid_size is enlarged to $r_search, so that the neighbor search finds all pairs within the cutoff of the distorted positions."
        grid_size = r_search
    end
    point_grid = PointGrid(rs_ion, frac_to_cart(Rs, lattice), grid_size=grid_size)
    ions = get_ions(rs_ion, ion_types, δrs_ion)

    return Structure(system, lattice, Rs, ions, point_grid)
end

function Structure(Rs::M1, rs_ion::M2, ion_types, lattice::M3, conf::Config=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf), system=get_system(conf)) where {M1,M2,M3<:Matrix{Float64}}
    return Structure(Rs, rs_ion, zeros(3, size(rs_ion, 2)), ion_types, lattice, conf; rcut=rcut, grid_size=grid_size, system=system)
end

"""
    check_cutoffs(conf=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf))

Error if neighbors within a cutoff would be missed: the neighbor grid only searches adjacent grid cells, so
`grid_size` must be at least `rcut`, and the translations and the grid of a structure are built for `rcut`, so the
`rcut` of the `ML` block must not be larger. With `rcut_pair_selection = "frame"`, `grid_size` is instead enlarged
to [`rcut_search`](@ref) if it is smaller (with a warning). With `rcut_pair_selection = "frame"` and an ML model,
`apply_distortion` and `apply_distance_distortion` of the `ML` block must be `true` (the descriptor would otherwise
use reference distances or angles with the frame pair set).
"""
function check_cutoffs(conf=get_empty_config(); rcut=get_rcut(conf), grid_size=get_grid_size(conf))
    grid_size ≥ rcut || error("grid_size = $grid_size is smaller than rcut = $rcut, neighbors would be missed. " *
                              "Remove grid_size or set it to at least rcut.")
    if get_ml_model(conf) && get_ml_rcut(conf) > rcut
        error("rcut = $(get_ml_rcut(conf)) of the ML block is larger than rcut = $rcut of the Options block, the ML " *
              "model would miss neighbors. Set rcut in the Options block to at least the ML rcut.")
    end
    if get_rcut_pair_selection(conf) == "frame" && get_ml_model(conf) &&
       !(get_ml_apply_distortion(conf) && get_ml_apply_distance_distortion(conf))
        error("rcut_pair_selection = \"frame\" requires apply_distortion = true and apply_distance_distortion = true " *
              "in the ML block (the ML descriptor would otherwise use reference distances or angles together with the " *
              "frame pair set).")
    end
end
