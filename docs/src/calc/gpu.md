# [GPU Support](@id gpu-support)

Hamster can build, diagonalize and differentiate Hamiltonians on NVIDIA GPUs. This is supported for standard runs and for the optimization of all Hamiltonian models (TB, ML kernel with and without `sim_mat`, SOC and Ewald onsites) with eigenvalue data.

## Installation

The GPU support is implemented as a package extension that is loaded together with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). CUDA.jl is a weak dependency of Hamster: it is never installed or loaded on CPU-only machines. To use a GPU, install CUDA.jl (and optionally [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) for the shift-invert eigensolver) into your default Julia environment:

```bash
julia -e 'using Pkg; Pkg.add(["CUDA", "CUDSS"])'
julia -e 'using CUDA; CUDA.versioninfo()'   # should list your GPU
```

If Hamster was installed before GPU support was added, update its environment once (e.g., with `hamster_update.jl` or `julia --project=/path/to/Hamster.jl -e 'using Pkg; Pkg.update()'`). Otherwise, an outdated version of a package shared with CUDA.jl (GPUArraysCore) may be loaded, which results in errors such as ``UndefVarError: `neutral_element` not defined``.

With CUDSS.jl installed, typical GPU runs are compiled when the extension is precompiled (on the first `using CUDSS` after an installation or update), which saves 20-30 s at the start of every run. This needs a GPU: on clusters, precompile on a GPU node (e.g., in a short job running `julia -e 'using Hamster, CUDA, CUDSS'`), not on a login node without GPU. Otherwise, every run compiles the GPU code paths on first use.

CUDA.jl downloads the CUDA runtime (several GB) into `~/.julia` when it is first used. On systems with a small home quota, move the Julia depot to a larger disk, or use a CUDA toolkit installed on the system by adding `[CUDA_Runtime_jll]` with `local = "true"` to the `LocalPreferences.toml` of your default environment.

## Usage

Set the `device` tag in the `Options` block:

```bash
begin Options
    device = gpu
end
```

The `hamster` executable then loads CUDA.jl (and CUDSS.jl, if available) automatically. When Hamster is used as a library, load the extension with `using CUDA` (and `using CUDSS`) before constructing the `EffectiveHamiltonian`.

With MPI, every rank binds to one GPU of its node in a round-robin fashion (node-local rank modulo number of GPUs). Typically, one MPI rank per GPU gives the best performance.

## What runs on the GPU

All models are linear in their parameters. When the `EffectiveHamiltonian` is constructed, Hamster precomputes for each structure the linear map from the model parameters to the nonzero real-space matrix elements and moves it to the GPU. During a calculation, the GPU then

1. evaluates the real-space Hamiltonian for the current parameters,
2. assembles the sparse k-space Hamiltonians,
3. diagonalizes them:
   - `sp_diag = false`: dense eigensolver of cuSOLVER (batched over k-points for matrices up to 32 × 32),
   - `sp_diag = true` and `diag_method = krylov-schur`: Krylov-Schur (KrylovKit.jl) with cuSPARSE matrix-vector products,
   - `sp_diag = true` and `diag_method = shift-invert`: Krylov-Schur on the shift-inverted Hamiltonian, factorized with cuDSS (requires CUDSS.jl),
4. computes the gradient of the loss w.r.t. the model parameters (Hellmann-Feynman theorem).

The optimizer, the loss and the MPI communication remain on the CPU, since they only involve eigenvalues and parameter vectors.

## Notes

- The precomputed linear maps of all structures stay in GPU memory for the whole run. If the GPU memory is exhausted, distribute the structures over more MPI ranks / GPUs.
- With the sparse eigensolver, the `neig` eigenvalues closest to `eig_target` are returned in ascending order.
- Fitting to real-space Hamiltonian data (`hr_fit`) and writing Hamiltonians (`write_hr`, `write_hk`) work with `device = gpu`, but these steps are performed on the CPU.
