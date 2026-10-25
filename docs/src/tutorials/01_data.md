# [1. Preparing DFT data](@id tutorial-data)

Hamster models are fitted to eigenvalues (band energies) from DFT; no wave functions or projections are needed. This
page describes which data a model needs and the file formats Hamster reads. The tutorials use data in these formats,
in [`tutorials/data`](https://github.com/TheoFEM-TUM/Hamster.jl/tree/main/tutorials/data). Any DFT code can be used:
convert its output to these formats with a small script of your own (e.g., with
[HDF5.jl](https://github.com/JuliaIO/HDF5.jl) or [h5py](https://www.h5py.org)).

## What to compute

| Data | Used for | Typical DFT calculation |
|---|---|---|
| Structure of the primitive cell | every run | relaxed structure |
| Band structure of the primitive cell | TB fit (and SOC fit, with SOC) | non-self-consistent run on a k-point path and/or a k-point grid |
| Band structure at other k-points | validation of the TB fit | as above, other k-points |
| Snapshots of an MD run | ML and electrostatics fits, validation | eigenvalues of every snapshot on a small k-point grid (Γ and a few more) |

Use the same functional and settings for all data of one model. The eigenvalues are used as given by the DFT code, in
eV, without shifting them to the valence band maximum; all data must therefore have a common energy reference (as is
the case for snapshots of the same supercell computed with the same settings).

The DFT calculation usually contains more bands than the model, e.g., semicore states below and high-lying states
above the bands of interest. The tag `bandmin` sets the DFT band that corresponds to the lowest model band; the model
has as many bands as orbitals (twice as many with SOC). For a supercell of ``N`` primitive cells, ``N`` times as many
bands of both the DFT data and the model have to be counted.

## Structures

- **POSCAR** (VASP format): primitive cell (`poscar` tag of the `Options` block) and reference structure of the
  snapshots (`poscar` tag of the `Supercell` block). The reference structure defines the species and the order of the
  atoms of all snapshots.
- **Trajectory** (`xdatcar` tag of the `Supercell` block): a VASP XDATCAR, or an HDF5 file with the datasets

  | Dataset | Shape (Julia) | Content |
  |---|---|---|
  | `configs` (or `positions`) | 3 × atoms × structures | fractional coordinates |
  | `lattice` | 3 × 3, or 3 × 3 × structures | lattice vectors (Å) as columns |

## Eigenvalues

- **EIGENVAL** (VASP format): one structure, e.g., the band structure of the primitive cell (`train_data`, `val_data`
  of the `Optimizer` block with `train_mode = pc`). The k-points of an EIGENVAL file can also be used for a band
  structure calculation (`kpoints` tag).
- **HDF5 file** for many structures (`train_mode = md`, `val_mode = md`), with the datasets

  | Dataset | Shape (Julia) | Content |
  |---|---|---|
  | `eigenvalues` | bands × k-points × structures | eigenvalues (eV), sorted for every k-point |
  | `kpoints` | 3 × k-points, or 3 × k-points × structures | k-points in fractional coordinates of the reciprocal lattice |

  The structure index of the eigenvalues must match the index of the trajectory. Additional datasets (e.g.,
  occupations) are ignored.

!!! note "Array order"
    Julia stores arrays in column-major order, Python (h5py, NumPy) in row-major order. An array written from Julia
    with shape (bands, k-points, structures) appears in Python (and in `h5ls`) with the shape (structures, k-points,
    bands), and vice versa. When you write the files from Python, use the reversed shapes, e.g.,
    `eigenvalues` with shape (structures, k-points, bands) and `configs` with shape (structures, atoms, 3).

A minimal example in Julia:

```julia
using HDF5
# E: bands × k-points × structures (eV), kp: 3 × k-points (fractional),
# X: 3 × atoms × structures (fractional), L: 3 × 3 (lattice vectors as columns, Å)
h5write("eigenval.h5", "eigenvalues", E)
h5write("eigenval.h5", "kpoints", kp)
h5write("structures.h5", "configs", X)
h5write("structures.h5", "lattice", L)
```

## The data of the tutorials

| File | Content |
|---|---|
| `gaas/POSCAR`, `gaas/SC_POSCAR` | primitive cell and 8-atom cubic cell of GaAs |
| `gaas/EIGENVAL_train`, `EIGENVAL_val`, `EIGENVAL_bands` | band structures of the primitive cell: training, validation, k-point path |
| `gaas/structures.h5`, `gaas/eigenval.h5` | 29 MD snapshots of the 8-atom cell and their eigenvalues |
| `gaas/params_tb.dat`, `gaas/ml_params.dat`, `gaas/rllm.dat` | fitted TB and ML parameters, tabulated distance dependence |
| `cspbbr3/POSCAR`, `cspbbr3/SC_POSCAR` | cubic cell and 2×2×2 supercell of CsPbBr₃ |
| `cspbbr3/EIGENVAL_bands`, `EIGENVAL_soc` | band structures without and with SOC (k-point path) |
| `cspbbr3/structures_md.h5`, `eigenval_md.h5` | 20 MD snapshots of the 2×2×2 supercell and their eigenvalues (without SOC) |
| `cspbbr3/params_tb.dat`, `cspbbr3/rllm.dat` | fitted TB parameters, tabulated distance dependence |

The files `rllm.dat` contain the distance dependence of the matrix elements on a grid (`load_rllm = true`,
`rllm_file`); without them, Hamster computes it at the start of a run, which takes longer. A file belongs to the
radial functions (`n`, `alpha`) and the `rcut` it was computed with; do not load it after changing these.
