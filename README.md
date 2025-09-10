# Hamster.jl

[![CI](https://github.com/mschwade-code/Hamster.jl/actions/workflows/runtests.yaml/badge.svg)](https://github.com/mschwade-code/Hamster.jl/actions/workflows/runtests.yaml)
[![codecov](https://codecov.io/gh/TheoFEM-TUM/Hamster.jl/graph/badge.svg?token=8MW6VZYIE2)](https://codecov.io/gh/TheoFEM-TUM/Hamster.jl)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://theofem-tum.github.io/Hamster.jl/dev/)

<p align="center">
  <img width="250" height="250" src="docs/src/assets/logo.png">
</p>

**H**amiltonian-learning **A**pproach for **M**ultiscale **S**imulations using a **T**ransferable and **E**fficient **R**epresentation

`Hamster.jl` is a pure-Julia package for fitting and running physics-informed effective Hamiltonian models to study temperature-dependent optoelectronic properties. Originally created by Martin Schwade and developed by the [TheoFEM group](https://theofem.de/) at TU Munich (Prof. D. A. Egger), it implements a Δ-machine-learning approach to correct tight-binding Hamiltonians in response to changes in the atomic environment. Spin–orbit coupling (SOC) is supported.

📖 [Documentation](https://theofem-tum.github.io/Hamster.jl/dev/)

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

While keyword arguments can be passed directly via the command line, it is more practical to provide Hamster with a config file `hconf`. Examples can be found [here](https://theofem-tum.github.io/Hamster.jl/dev/examples/examples/).

## License
This project is licensed under the MIT License.

## Citing
If you use `Hamster.jl` in your work, please cite our work.  
A list of relevant publications is provided below.

## Publications

[1] [M. Schwade, M. J. Schilcher, C. Reverón Baecker, M. Grumet, and D. A. Egger, J. Chem. Phys. 160, 134102 (2024)](https://pubs.aip.org/aip/jcp/article/160/13/134102/3280389/Temperature-transferable-tight-binding-model-using)

[2] [M. Schwade, S. Zhang, F. Vonhoff, F. P. Delgado, D. A. Egger, arXiv:2508.20536 [cond-mat.mtrl-sci] (2025)](https://arxiv.org/abs/2508.20536)