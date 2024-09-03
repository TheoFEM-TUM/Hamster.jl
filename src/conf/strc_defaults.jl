"""
    rcut=7.

The parameter `rcut` sets the cut-off radius beyond which all interactions are neglected. A cut-off radius of zero means all interactions are considered.
"""
get_rcut(conf::Config)::Float64 = conf("rcut") == "default" ? 7. : conf("rcut")

"""
    poscar=POSCAR

The parameter `poscar` sets the path to the POSCAR VASP file that defines the base system.
"""
get_poscar(conf::Config)::String = conf("poscar") == "default" ? "POSCAR" : conf("POSCAR")

"""
    grid_size=rcut

The parameter `grid_size` determines the size of the cubes that are used to divide the simulation cell.
"""
get_grid_size(conf::Config)::Float64 = conf("grid_size") == "default" ? get_rcut(conf) : conf("grid_size")