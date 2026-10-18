# Hamster.jl

**H**amiltonian-learning **A**pproach for **M**ultiscale **S**imulations using a **T**ransferable and **E**fficient **R**epresentation

`Hamster.jl` is a pure-Julia package for fitting and running effective Hamiltonians to study temperature-dependent optoelectronic properties. Originally created by Martin Schwade and developed by the [TheoFEM group](https://theofem.de/) at TU Munich (Prof. D. A. Egger), it implements a Δ-machine-learning approach to correct tight-binding Hamiltonians in response to changes in the atomic environment. Spin–orbit coupling (SOC) is supported.

## Installation

Since `Hamster.jl` is not (yet) a registered Julia package, it comes with the `hamster` executable, which installs the dependencies and runs calculations:

```bash
git clone https://github.com/TheoFEM-TUM/Hamster.jl.git
cd Hamster.jl
./hamster --install [--add_path yes/no] [--bashrc FILE] [--add_test_exec] [--test] [--gpu]
```

By default, the `Hamster.jl` folder is added to `PATH` in your `.bashrc`. `--add_test_exec` creates a `hamster_test` executable for the test suite, `--test` runs the test suite after the installation. `--gpu` also installs CUDA.jl and CUDSS.jl for GPU runs; the GPU code is precompiled only if a GPU is available, otherwise precompile it once on a GPU node with `julia -e 'using Hamster, CUDA, CUDSS'`.

## Command-line interface

```bash
hamster [-t THREADS] [--conf FILE] [--TAG VALUE ...]   # run a calculation
hamster --install [options]                           # install the dependencies and precompile Hamster
hamster --update [--no-pull]                          # pull the latest code (git pull --ff-only), update the dependencies and precompile
hamster --version                                     # version of Hamster (and the git commit) and Julia
hamster --help                                        # overview of all commands and options
```

## Quickstart

You can start Hamster by calling the `hamster` executable. To make use of MPI parallelization, add `mpiexecjl` or `srun` in front.

```bash
[mpiexecjl -n NRANKS / srun] hamster [-t NUMBER_OF_THREADS] [kwargs]
```

While keyword arguments can be passed directly via the command line, it is more practical to provide Hamster with a [config file](@ref man-config) `hconf`. Examples can be found [here](@ref examples).

