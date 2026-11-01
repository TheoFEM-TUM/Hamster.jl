# [Standard Run](@id standard-run)

```@docs
Hamster.run_calculation(::Val{:standard}, ::Any, ::Config)
```

## Dense diagonalization

With `sp_diag = false` (default), all eigenvalues are computed with LAPACK. The solver is chosen automatically:
Hamiltonians that are real (at Γ and the other time-reversal invariant k-points without SOC) are diagonalized in real
arithmetic, which is several times faster; for large complex Hamiltonians, the MRRR solver is used. Small Hamiltonians
are diagonalized in parallel over the k-points, large ones one after the other with all threads in BLAS. For large
supercells on Intel CPUs, install Hamster with `--mkl` (see the [command-line interface](@ref man-cli)) to use Intel MKL instead of OpenBLAS.
