"""
    rcut=7.

The parameter `rcut` sets the cut-off radius beyond which all interactions are neglected. A cut-off radius of zero means all interactions are considered.
"""
get_rcut(conf::Config)::Float64 = get(conf, "rcut", 7.0)

"""
    rcut_tol=1.

The parameter `rcut_tol` sets a tolerance for the cutoff function.
"""
get_rcut_tol(conf::Config)::Float64 = get(conf, "rcut_tol", 1.0)

"""
    Rmax=1

The parameter `Rmax` sets the maximum magnitude of a translation vector used for periodic boundaty conditions. Note that this is determined automatically if `rcut` is set.
"""
get_Rmax(conf::Config)::Int64 = conf("Rmax") == "default" ? 1 : conf("Rmax")

"""
    poscar=POSCAR

The parameter `poscar` sets the path to the POSCAR VASP file that defines the base system.
"""
get_poscar(conf::Config)::String = conf("poscar") == "default" ? "POSCAR" : conf("poscar")

"""
    grid_size=rcut

The parameter `grid_size` determines the size of the cubes that are used to divide the simulation cell.
"""
get_grid_size(conf::Config)::Float64 = conf("grid_size") == "default" ? get_rcut(conf) : conf("grid_size")