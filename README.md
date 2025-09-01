# Hamster.jl

[![CI](https://github.com/mschwade-code/Hamster.jl/actions/workflows/runtests.yaml/badge.svg)](https://github.com/mschwade-code/Hamster.jl/actions/workflows/runtests.yaml)
[![codecov](https://codecov.io/gh/mschwade-code/Hamster.jl/graph/badge.svg?token=8MW6VZYIE2)](https://codecov.io/gh/mschwade-code/Hamster.jl)

<p align="center">
  <img width="250" height="250" src="docs/src/assets/logo.png">
</p>


Hamster.jl is a powerful Julia package to fit and run calculations with effective Hamiltonians to compute temperature-dependent optoelectronic properties.

## Installation

Since `Hamster.jl` is not (yet) a registered Julia package, we provide an installation script that sets up dependencies, sets the PATH variable and creates the `hamster` executable.

```bash
julia hamster_install.jl [--add_path yes/no] [--exec_name hamster] [--bashrc default] [--add_test_exec]
```

## Running Hamster

You can run start Hamster by calling the `hamster` executable. To make use of MPI parallelization you need add `mpiexecjl` or `srun` in front.

```bash
[mpiexecjl -n NODES / srun] hamster [kwargs]
```

While keyword arguments can be passed directly via the command line, it is more practical to provide Hamster with a config file `hconf` (see examples).

## Publications

[1] [M. Schwade, M. J. Schilcher, C. Reverón Baecker, M. Grumet, and D. A. Egger, J. Chem. Phys. 160, 134102 (2024)](https://pubs.aip.org/aip/jcp/article/160/13/134102/3280389/Temperature-transferable-tight-binding-model-using)

[2] [M. Schwade, S. Zhang, F. Vonhoff, F. P. Delgado, D. A. Egger, arXiv:2508.20536 [cond-mat.mtrl-sci] (2025)](https://arxiv.org/abs/2508.20536)