const DENSE_LARGE = 1000
const THREADS_PER_KPOINT = Ref(0)

"""
    set_threads_per_kpoint!(n)

Set the number of threads per k-point of all loops over k-points (`n = 0`: automatic, see `threads_per_kpoint`).
"""
set_threads_per_kpoint!(n::Integer) = (THREADS_PER_KPOINT[] = n; nothing)

"""
    threads_per_kpoint(Nε, nitems; dense=true, n=THREADS_PER_KPOINT[]) -> Int

Threads per k-point for a loop over `nitems` k-points with Hamiltonians of `Nε` orbitals: `n` if set (at most the
number of Julia threads), otherwise automatic: all threads for dense Hamiltonians with at least `DENSE_LARGE` orbitals,
one for smaller dense ones, and the threads left over by the k-points (number of threads ÷ `nitems`) for sparse ones.
"""
function threads_per_kpoint(Nε, nitems; dense=true, n=THREADS_PER_KPOINT[])
    T = Threads.nthreads()
    n > 0 && return min(n, T)
    dense && return Nε ≥ DENSE_LARGE ? T : 1
    return max(1, T ÷ max(nitems, 1))
end

"""
    kpoint_tmap(f, items, Nε; dense=true, n=threads_per_kpoint(Nε, length(items), dense=dense))

Map `f` over `items` (the k-points of one structure) with `n` threads per item: `nthreads ÷ n` items at a time, each
with `n` threads for its inner loops (`inner_tforeach`) and for Intel MKL. OpenBLAS has one thread count for all calls:
it gets all threads when the items run one after the other (`n = nthreads`), otherwise one.
"""
function kpoint_tmap(f, items, Nε; dense=true, n=threads_per_kpoint(Nε, length(items), dense=dense))
    isempty(items) && return map(f, items)
    T = Threads.nthreads()
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(n ≥ T ? T : 1)
    try
        n ≥ T && return with_item_threads(() -> map(f, items), T)
        return tmap(x -> with_item_threads(() -> f(x), n), items; ntasks=min(length(items), T ÷ n))
    finally
        BLAS.set_num_threads(blas_threads)
    end
end

"""
    with_item_threads(f, n)

Call `f()` with `n` tasks for the inner loops and `n` MKL threads.
"""
with_item_threads(f, n) = with_mkl_threads(() -> with_inner_tasks(f, n), n)

"""
    inner_tasks()

Number of tasks for the inner loops of the current item of `kpoint_tmap` (the number of Julia threads outside of it).
"""
inner_tasks() = get(task_local_storage(), :hamster_inner_tasks, Threads.nthreads())

"""
    with_inner_tasks(f, n)

Call `f()` with `inner_tasks() == n`.
"""
with_inner_tasks(f, n) = task_local_storage(f, :hamster_inner_tasks, n)

"""
    inner_tforeach(f, range; chunks_per_task=1)

Call `f` in parallel on `chunks_per_task * inner_tasks()` parts of `range` (more than one part per task for load
balancing), or once on the whole `range` with a single task.
"""
function inner_tforeach(f, range; chunks_per_task=1)
    n = inner_tasks()
    (n == 1 || length(range) ≤ 1) && return (f(range); nothing)
    tforeach(f, chunks(range, n=min(length(range), chunks_per_task * n)))
    return nothing
end

"""
    blas_library()

Path of the BLAS/LAPACK library that Julia uses (OpenBLAS, or Intel MKL after `hamster --install --mkl`).
"""
blas_library() = first(BLAS.get_config().loaded_libs).libname

"""
    uses_mkl()

Whether Julia uses Intel MKL for BLAS/LAPACK.
"""
uses_mkl() = occursin("mkl", basename(blas_library()))

"""
    with_mkl_threads(f, n)

Call `f()` with `n` MKL threads for the calling Julia thread; without Intel MKL, just call `f()`. MKL ignores
`BLAS.set_num_threads` for calls from Julia threads and uses `MKL_NUM_THREADS` threads in each of them.
"""
function with_mkl_threads(f, n)
    uses_mkl() || return f()
    setlocal = Libc.Libdl.dlsym(Libc.Libdl.dlopen(blas_library()), :MKL_Set_Num_Threads_Local)
    previous = ccall(setlocal, Cint, (Cint,), n)
    try
        return f()
    finally
        ccall(setlocal, Cint, (Cint,), previous)
    end
end
