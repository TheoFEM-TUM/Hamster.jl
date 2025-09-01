# Hamster.jl

**H**amiltonian-learning **A**pproach for **M**ultiscale **S**imulations using a **T**ransferable and **E**fficient **R**epresentation

`Hamster.jl` is a pure-Julia package for fitting and running effective Hamiltonians to study temperature-dependent optoelectronic properties. Originally created by Martin Schwade and developed by the [TheoFEM group](https://theofem.de/) at TU Munich (Prof. D. A. Egger), it implements a Δ-machine-learning approach to correct tight-binding Hamiltonians in response to changes in the atomic environment. Spin–orbit coupling (SOC) is supported.

## Installation

Since `Hamster.jl` is not (yet) a registered Julia package, we provide an installation script that sets up dependencies, sets the PATH variable and creates the `hamster` executable.

```bash
julia hamster_install.jl [--add_path yes/no] [--exec_name hamster] [--bashrc default] [--add_test_exec]
```

## Quickstart

You can run start Hamster by calling the `hamster` executable. To make use of MPI parallelization you need add `mpiexecjl` or `srun` in front.

```bash
[mpiexecjl -n NODES / srun] hamster [kwargs]
```

While keyword arguments can be passed directly via the command line, it is more practical to provide Hamster with a [config file](@ref man-config) `hconf`. Examples can be found [here](@ref examples).

