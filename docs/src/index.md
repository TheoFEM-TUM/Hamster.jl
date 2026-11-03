# Hamster.jl

**H**amiltonian-learning **A**pproach for **M**ultiscale **S**imulations using a **T**ransferable and **E**fficient **R**epresentation

`Hamster.jl` is a pure-Julia package for fitting and running effective Hamiltonians to study temperature-dependent optoelectronic properties. Originally created by Martin Schwade and developed by the [TheoFEM group](https://theofem.de/) at TU Munich (Prof. D. A. Egger), it implements a Δ-machine-learning approach to correct tight-binding Hamiltonians in response to changes in the atomic environment. Spin–orbit coupling (SOC) is supported.

## Installation

Since `Hamster.jl` is not (yet) a registered Julia package, it comes with the `hamster` executable, which installs the dependencies and runs calculations:

```bash
git clone https://github.com/TheoFEM-TUM/Hamster.jl.git
cd Hamster.jl
./hamster --install [--add_path yes/no] [--bashrc FILE] [--add_test_exec] [--test] [--gpu] [--mkl]
```

By default, the `Hamster.jl` folder is added to `PATH` in your `.bashrc`. `--add_test_exec` creates a `hamster_test` executable for the test suite, `--test` runs the test suite after the installation. `--mkl` also installs MKL.jl (Intel MKL for BLAS/LAPACK, faster dense diagonalization on Intel CPUs). `--gpu` also installs CUDA.jl and CUDSS.jl for GPU runs; the GPU code is precompiled only if a GPU is available, otherwise run `hamster --precompile` once on a GPU node.

## Command-line interface

```bash
hamster [-t THREADS] [--conf FILE] [--TAG VALUE ...]   # run a calculation
hamster --install [options]                           # install the dependencies and precompile Hamster
hamster --update [--no-pull]                          # pull the latest code (git pull --ff-only), update the dependencies and precompile
hamster --precompile                                  # precompile Hamster (and its GPU code on a GPU node)
hamster --version                                     # version of Hamster (and the git commit) and Julia
hamster --help                                        # overview of all commands and options
```

See [Command-Line Interface](@ref man-cli) for all commands and options.

## Quickstart

You can start Hamster by calling the `hamster` executable. To make use of MPI parallelization, add `mpiexecjl` or `srun` in front.

```bash
[mpiexecjl -n NRANKS / srun] hamster [-t NUMBER_OF_THREADS] [kwargs]
```

While keyword arguments can be passed directly via the command line, it is more practical to provide Hamster with a [config file](@ref man-config) `hconf`. Examples can be found [here](@ref examples).

## [Parallelization](@id parallelization)

Hamster combines MPI and threads:

- **MPI ranks** share the structures of a run (and of the training and validation sets of a fit). This is the most
  efficient level, in particular for fits with many small structures: use as many ranks as the memory allows, with a
  few threads each.
- **Threads** work within one structure. Steps without k-points (neighbor lists, geometry tensor, the real-space
  Hamiltonian with TB, ML, SOC and electrostatic terms, the gradients of the model parameters) always use all threads.
  Steps over k-points (the Hamiltonians H(k), the diagonalization, the gradient per k-point and the exact DOS) use the
  same rule everywhere, set with `threads_per_kpoint` in the `Options` block:
  - `threads_per_kpoint = 1`: all k-points in parallel, one thread each;
  - `threads_per_kpoint = n`: `nthreads ÷ n` k-points at a time with `n` threads each;
  - `threads_per_kpoint = auto` (default): one thread per k-point for dense Hamiltonians with fewer than 1000 orbitals,
    all threads per k-point (one k-point after the other) for larger ones, and for sparse Hamiltonians the threads
    that are left over when there are fewer k-points than threads.

  The threads of a k-point are used by Hamster's own loops and by BLAS/LAPACK. OpenBLAS has a single thread count for
  all calls, so it uses one thread when several k-points run at the same time and at most 32 threads otherwise (with
  all cores of a node, single OpenBLAS calls can take many times longer); Intel MKL (`hamster --install --mkl`)
  uses `n` threads per k-point.

