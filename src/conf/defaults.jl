# ====================
# Input/Output
# ====================
@configtag verbosity Int64 1 "Controls output verbosity (0=minimal, 1=normal, 2=verbose, 3=debug)."
@configtag write_current Bool false "Whether to write current operator."
@configtag current_file  String "ham.h5" "File where current operator is stored."
@configtag write_hk      Bool false "Whether to write k-space Hamiltonians."
@configtag ham_file      String "ham.h5" "File where Hamiltonians are stored."
@configtag write_hr      Bool get_write_current(conf) "Whether to write real-space Hamiltonians. Defaults to true if write_current=true."
@configtag kpoints String "gamma" "file/method for defining k-points (gamma, EIGENVAL, filename.h5)."
@configtag save_vecs Bool false "Whether eigenvectors are stored."


# ====================
# Model setup
# ====================
@configtag system String "unknown" "name of the system"
@configtag init_params String "ones" "TB parameter initialization (file, ones, zeros, rand)."


# ====================
# Diagonalization
# ====================
@configtag skip_diag Bool false "if true, no eigenvalues are computed."
@configtag diag_method String "shift-invert" "sparse eigensolver when `sp_diag=true` (shift-invert, krylov-schur)."
@configtag neig Int64 6 "number of eigenvalues when using sparse eigensolver."
@configtag eig_target Float64 0 "target energy when using sparse eigensolver."


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