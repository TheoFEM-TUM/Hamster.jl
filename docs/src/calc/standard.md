# [Standard Run](@id standard-run)

```@docs
Hamster.run_calculation(::Val{:standard}, ::Any, ::Config)
```

## Dense diagonalization

With `sp_diag = false` (default), all eigenvalues are computed with LAPACK. The solver is chosen automatically:
Hamiltonians that are real (at Γ and the other time-reversal invariant k-points without SOC) are diagonalized in real
arithmetic, which is several times faster; for large complex Hamiltonians, the MRRR solver is used with OpenBLAS. Small Hamiltonians
are diagonalized in parallel over the k-points, large ones one after the other with all threads in BLAS (see
[Parallelization](@ref parallelization)). For large supercells on Intel CPUs, install Hamster with `--mkl` (see the
[command-line interface](@ref man-cli)) to use Intel MKL instead of OpenBLAS.

## Sparse diagonalization

For large supercells, only a few eigenvalues are usually needed, e.g. the band edges. With `sp_diag = true`, the
k-space Hamiltonians stay sparse and a sparse eigensolver computes the `neig` eigenvalues closest to `eig_target`
(returned in ascending order). Place `eig_target` in the band gap to obtain the states at the band edges. The solver is
chosen with `eigsolver` (formerly `diag_method`, which is still accepted):

- `shift-invert` (default): Krylov-Schur on (H - σ)⁻¹, with a sparse LU factorization (UMFPACK via Arpack on the CPU,
  cuDSS on the GPU). Robust and fast for small and medium systems; the memory of the factorization grows faster than
  linearly with the system size.
- `chebyshev`: Chebyshev-filtered subspace iteration on (H - σ)². Only products of H with blocks of vectors are needed,
  so the memory grows linearly with the system size and the solver is threaded on the CPU and runs on GPUs
  (`device = gpu`). Recommended for large supercells, in particular for few eigenvalues.
- `krylov-schur`: Krylov-Schur on H itself (KrylovKit). Converges slowly for eigenvalues inside the spectrum and can
  miss some of them; mainly useful for eigenvalues at the edges of the spectrum.

The Chebyshev solver is implemented in [`Hamster.chebyshev_eigen`](@ref).
