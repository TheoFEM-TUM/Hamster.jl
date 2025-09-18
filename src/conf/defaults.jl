"""
**verbosity**=1

The `verbosity` parameter controls the amount of output printed to the console or written to output files.  
A value of `0` disables most print statements.

Possible options:
- `0`: minimal output; most print statements are suppressed.
- `1`: normal output volume (default).
- `2`: increased output; additional information is printed.
- `3`: maximum output; includes detailed debug information.
"""
get_verbosity(conf::Config)::Int64 = conf("verbosity") == "default" ? 1 : conf("verbosity")

"""
**system**=unknown

The `system` tag gives a name to the system under study.
"""
get_system(conf::Config)::String = conf("system") == "default" ? "unknown" : conf("system")

"""
**init_params** = ones

The `init_params` tag determines how the tight-binding parameters are initialized.  

Possible options:
- `filename.dat`: initialize from a file named `filename.dat`.
- `ones`: initialize all parameters to 1 (default).
- `zeros`: initialize all parameters to 0.
- `rand`: initialize parameters randomly between 0 and 1.
"""
get_init_params(conf::Config)::String = conf("init_params") == "default" ? "ones" : conf("init_params")

"""
**kpoints**=gamma

The `kpoints::String` tag specifies the file or method used to define the set of k-points.

Possible options:
- `gamma`: use only the Gamma point (default).
- `EIGENVAL`: read k-points from a VASP `EIGENVAL` file.
- `filename.h5`: read k-points from the `"k-points"` field of an HDF5 file.
"""
get_kpoints_file(conf::Config)::String = conf("kpoints") == "default" ? "gamma" : conf("kpoints")

"""
**neig**=6 (only if sp_diag=true)

The `neig::Int` tag sets the number of eigenvalues that are calculated when using a sparse eigensolver.
"""
get_neig(conf::Config)::Int64 = conf("Neig") == "default" ? 6 : conf("Neig")

"""
**save_vecs**=false

The `save_vecs::Bool` tag determines whether the eigenvectors are written to a file.
"""
get_save_vecs(conf::Config)::Bool = conf("save_vecs") == "default" ? false : conf("save_vecs")

"""
**write_hk**=false

The `write_hk::Bool` tag determines whether the Hamiltonians in k-space are written to a file.
"""
get_write_hk(conf::Config)::Bool = conf("write_hk") == "default" ? false : conf("write_hk")

"""
**eig_target**=0.

The `eig_target::Float` tag sets the target energy when using a sparse eigensolver.
"""
get_eig_target(conf::Config)::Float64 = conf("eig_target") == "default" ? 0. : conf("eig_target")

"""
**diag_method**=shift-invert

The `diag_method` tag sets the method to be used for calculating eigenvalues when `sp_diag=true`, ignored otherwise.

Possible options:
- `shift-invert` (default): call `eigs` function from `Arpack`.
- `krylov-schur`: call `eigsolve` function from `KrylovKit`.
"""
get_diag_method(conf::Config)::String = conf("diag_method") == "default" ? "shift-invert" : conf("diag_method")

"""
**skip_diag**=false

If `skip_diag` is set to true, no eigenvalues are computed for the Hamiltonian.
"""
get_skip_diag(conf::Config)::Bool = conf("skip_diag") == "default" ? false : conf("skip_diag")

"""
**nthreads_kpoints**=JULIA_NUM_THREADS

The `nthreads_kpoints::Int` tag sets the number of tasks to work on kpoints simultaneously.
"""
get_nthreads_kpoints(conf::Config)::Int64 = conf("nthreads_kpoints") == "default" ? Threads.nthreads() : conf("nthreads_kpoints") 

"""
**nthreads_bands**=JULIA_NUM_THREADS

The `nthreads_bands::Int` tag sets the number of tasks to work on energy bands simultaneously.
"""
get_nthreads_bands(conf::Config)::Int64 = conf("nthreads_bands") == "default" ? Threads.nthreads() : conf("nthreads_bands")

"""
**nthreads_blas**=1

The `nthreads_blas::Int` tag sets the number of threads used by the BLAS library.
"""
get_nthreads_blas(conf::Config)::Int64 = conf("nthreads_blas") == "default" ? 1 : conf("nthreads_blas")

"""
    nhamster=1

The `nhamster` tag sets the number of `Hamster` processes to be spawned for parallel tasks.
"""
get_nhamster(conf::Config)::Int64 = conf("nhamster") == "default" ? 1 : conf("nhamster")

"""
    seed=none

The `seed` tag can be used to set a custom seed for RNG.
"""
function set_seed!(conf::Config; rank=0)
    if conf("seed") ≠ "default"
        Random.seed!(conf("seed") + rank)
    end
end