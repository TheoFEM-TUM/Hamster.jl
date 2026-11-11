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
@configtag model_file String "model.h5" "Path to HDF5 file for saving/loading complete model state (parameters + hyperparameters)." category="input/output"
@configtag load_model Bool false "Load model from model_file at startup, overwriting config tags with stored values." category="input/output" see_also=["model_file", "save_model"]
@configtag save_model Bool false "Export model parameters, settings and radial tables to model_file after initialization or, in a fit, after the final optimization step." category="input/output" see_also=["model_file", "load_model"]


# ====================
# Model setup
# ====================
@configtag system String "unknown" "name of the system" category="structure"
@configtag init_params String "ones" "TB parameter initialization (file, ones, zeros, rand)." category="model"


# ====================
# Diagonalization
# ====================
@configtag skip_diag Bool false "if true, no eigenvalues are computed." category="diagonalization"
@configtag eigsolver String "shift-invert" "sparse eigensolver when `sp_diag=true`: shift-invert (factorization, Arpack/cuDSS), krylov-schur (KrylovKit) or chebyshev (Chebyshev-filtered subspace iteration, only matrix-vector products, threaded and on GPUs). Formerly `diag_method`." category="diagonalization" see_also=["sp_diag", "neig", "eig_target"] values=["shift-invert", "krylov-schur", "chebyshev"]
@configtag neig Int64 6 "number of eigenvalues when using sparse eigensolver." category="diagonalization" see_also=["sp_diag", "eig_target"]
@configtag eig_target Float64 0.0 "target energy when using sparse eigensolver." category="diagonalization" see_also=["sp_diag", "neig"]


"""
**threads_per_kpoint**=auto

The `threads_per_kpoint` tag sets how the threads of every loop over k-points (H(k), diagonalization, fit gradient,
exact DOS) are used: `n` threads per k-point, with `nthreads ÷ n` k-points at a time. `1` diagonalizes all k-points in
parallel with one thread each, the number of Julia threads one k-point after the other with all threads. `auto`
(returned as 0) uses all threads per k-point for dense Hamiltonians with at least `DENSE_LARGE` orbitals and otherwise
gives the threads that are left over when there are fewer k-points than threads to the k-points.
"""
function get_threads_per_kpoint(conf::Config)::Int64
    n = conf("threads_per_kpoint")
    (n == "default" || lowercase(string(n)) == "auto") && return 0
    n isa Integer && n ≥ 1 || throw(ArgumentError("threads_per_kpoint must be auto or a positive integer, not $n."))
    return n
end

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
@configtag device String "cpu" "Compute device for building, diagonalizing and differentiating Hamiltonians (cpu, gpu). `gpu` requires `using CUDA` (and `using CUDSS` for the shift-invert eigensolver); each MPI rank binds to one GPU of its node." category="device and parallelization" values=["cpu", "gpu"]

# ====================
# Post-processing
# ====================
@configtag dos Vector{String} String[] "post-processing of a standard run: any combination of dos (total DOS), pdos (projected DOS), cohp (crystal orbital Hamilton populations) and jdos (joint DOS, needs nelec), or all (dos, pdos and cohp), e.g. `dos = pdos cohp`. Results are written to dos.h5; further settings in the DOS block." category="post-processing"

@configtag threads_per_kpoint String "auto" "threads per k-point in all loops over k-points (H(k), diagonalization, fit gradient, exact DOS): n threads per k-point and nthreads ÷ n k-points at a time, or auto (all threads per k-point for dense Hamiltonians with at least 1000 orbitals, otherwise the k-points in parallel)." getter=:custom value=(conf -> (n = get_threads_per_kpoint(conf); n == 0 ? "auto" : string(n))) category="device and parallelization"
@configtag nhamster Int64 1 "number of Hamster processes for parallel tasks." getter=:custom category="device and parallelization"
@configtag seed Int64 0 "seed of the random number generator (plus the MPI rank); without it, runs are not reproducible." getter=:custom value=(conf -> conf("seed") == "default" ? "none" : conf("seed")) default_doc="none" category="model"
