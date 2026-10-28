# ====================
# Input/Output
# ====================
@configtag verbosity Int64 1 "Controls output verbosity (0=minimal, 1=normal, 2=verbose, 3=debug)." category="input/output"
@configtag h5_deflate Int64 0 "Deflate compression level used for HDF5 datasets. Valid values are 0-9, where 0 disables compression, 1 gives the fastest compression, and 9 gives the highest compression effort." category="input/output"
@configtag write_current Bool false "Whether to write current operator." category="input/output" see_also=["current_file", "current_components", "write_hr"]
@configtag current_file  String "ham.h5" "File where current operator is stored." category="input/output" see_also=["write_current"]
@configtag current_components String "xyz" "Which Cartesian components of the current operator to compute and write, any combination of 'x', 'y', 'z' (e.g. \"x\" or \"xz\"). Defaults to all three." category="input/output" see_also=["write_current"]
@configtag write_hk      Bool false "Whether to write k-space Hamiltonians." category="input/output" see_also=["ham_file"]
@configtag ham_file      String "ham.h5" "File where Hamiltonians are stored." category="input/output"
@configtag write_hr      Bool get_write_current(conf) "Whether to write real-space Hamiltonians. Defaults to true if write_current=true." category="input/output" default_doc="true if write_current = true, else false" see_also=["ham_file"]
@configtag kpoints String "gamma" "file/method for defining k-points (gamma, EIGENVAL, filename.h5)." category="diagonalization"
@configtag save_vecs Bool false "Whether eigenvectors are stored." category="input/output"


# ====================
# Model setup
# ====================
@configtag system String "unknown" "name of the system" category="structure"
@configtag init_params String "ones" "TB parameter initialization (file, ones, zeros, rand)." category="model"


# ====================
# Diagonalization
# ====================
@configtag skip_diag Bool false "if true, no eigenvalues are computed." category="diagonalization"
@configtag diag_method String "shift-invert" "sparse eigensolver when `sp_diag=true` (shift-invert, krylov-schur)." category="diagonalization" see_also=["sp_diag", "neig", "eig_target"]
@configtag neig Int64 6 "number of eigenvalues when using sparse eigensolver." category="diagonalization" see_also=["sp_diag", "eig_target"]
@configtag eig_target Float64 0.0 "target energy when using sparse eigensolver." category="diagonalization" see_also=["sp_diag", "neig"]


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

# ====================
# Device
# ====================
@configtag device String "cpu" "Compute device for building, diagonalizing and differentiating Hamiltonians (cpu, gpu). `gpu` requires `using CUDA` (and `using CUDSS` for the shift-invert eigensolver); each MPI rank binds to one GPU of its node." category="device and parallelization"

# ====================
# Post-processing
# ====================
@configtag dos Vector{String} String[] "post-processing of a standard run: any combination of dos (total DOS), pdos (projected DOS), cohp (crystal orbital Hamilton populations) and jdos (joint DOS, needs nelec), or all (dos, pdos and cohp), e.g. `dos = pdos cohp`. Results are written to dos.h5; further settings in the DOS block." category="post-processing"

@configtag nthreads_kpoints Int64 Threads.nthreads() "number of tasks working on k-points simultaneously." getter=:custom default_doc="number of Julia threads" category="device and parallelization"
@configtag nthreads_bands Int64 Threads.nthreads() "number of tasks working on bands simultaneously." getter=:custom default_doc="number of Julia threads" category="device and parallelization"
@configtag nthreads_blas Int64 1 "number of BLAS threads (diagonalizations with fewer k-points than threads use more)." getter=:custom category="device and parallelization"
@configtag nhamster Int64 1 "number of Hamster processes for parallel tasks." getter=:custom category="device and parallelization"
@configtag seed Int64 0 "seed of the random number generator (plus the MPI rank); without it, runs are not reproducible." getter=:custom value=(conf -> conf("seed") == "default" ? "none" : conf("seed")) default_doc="none" category="model"
