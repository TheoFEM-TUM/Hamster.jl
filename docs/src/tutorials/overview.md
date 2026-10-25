# [Tutorials](@id tutorials)

These tutorials show how to build a Hamster model for your material, step by step: from the DFT data to a fitted
tight-binding (TB) model, spin-orbit coupling, machine-learned corrections for distorted structures, electrostatics,
and the application of the model to large cells. They assume a basic knowledge of DFT and band structures, but no
experience with tight-binding models or Hamster.

All tutorials run on a laptop within a few minutes. The inputs are in the
[`tutorials`](https://github.com/TheoFEM-TUM/Hamster.jl/tree/main/tutorials) folder of the repository, with one
folder per tutorial. Most tutorials come as an exercise: the config file `hconf` has gaps (`???`) that you fill in
with the help of the tutorial page and of `hamster --tag <name>`, which describes every input tag. The complete config
files are in the `solution` subfolder of each tutorial.

| Tutorial | Topic | Material | Type |
|---|---|---|---|
| [1. Preparing DFT data](@ref tutorial-data) | data formats, what to compute with DFT | – | reading |
| [2. A first run](@ref tutorial-first-run) | band structure from existing parameters | GaAs | example |
| [3. Fitting a TB model](@ref tutorial-tb-fit) | TB fit to a primitive cell | GaAs | exercise |
| [4. Hyperparameter optimization](@ref tutorial-hyperopt) | radial functions (`alpha`) | GaAs | example |
| [5. Spin-orbit coupling](@ref tutorial-soc) | SOC on top of a TB model | CsPbBr₃ | exercise |
| [6. ML corrections](@ref tutorial-ml) | Δ-ML model trained on MD snapshots | GaAs | exercise |
| [7. Electrostatics](@ref tutorial-electrostatics) | electrostatic on-site corrections (Wolf) | CsPbBr₃ | exercise |
| [8. Applications](@ref tutorial-applications) | large cells: band edges, DOS, PDOS, COHP | GaAs | example |

The page [Building a model](@ref tutorial-workflow) summarizes the steps, the building blocks of a Hamster model and
our recommendations, and is a good place to come back to when you fit your own material.

## Running the tutorials

Install Hamster (see [Installation](@ref man-cli)), then run each tutorial in its folder:

```bash
cd Hamster.jl/tutorials/02_first_run
hamster -t 4                 # 4 threads; reads the config file hconf in the current folder
julia --project=../.. plot.jl
```

Some tutorials have several steps; they come with a `run.sh` script that runs all of them (`./run.sh -t 4`, or
`../run.sh -t 4` in the `solution` folder). The paths in the config files are relative to the tutorial folder (or to
the `solution` folder); copy a tutorial to another place together with the `data` folder, or adapt the paths.

### Plots

The `plot.jl` scripts use the helper functions in
[`tutorials/plotting.jl`](https://github.com/TheoFEM-TUM/Hamster.jl/blob/main/tutorials/plotting.jl), which plot with
[Plots.jl](https://docs.juliaplots.org). Plots.jl is not a dependency of Hamster; install it once in your default Julia
environment:

```bash
julia -e 'using Pkg; Pkg.add("Plots")'
```

Without Plots.jl, the scripts write the data of every figure to a CSV file instead. All figures on these pages were made
with these scripts and the solutions; small differences to your results are expected, since the fits start from random
samples (e.g., of training structures and kernel points).

## Data and citation

The DFT data of the tutorials are small subsets of the data of our paper, which are available on
[Zenodo](https://doi.org/10.5281/zenodo.18485403):

- GaAs: band structures of the primitive cell and snapshots of an MD run (8-atom cell).
- CsPbBr₃: band structures of the cubic cell with and without SOC, and snapshots of an MD run (2×2×2 supercell).

The config files of the models in the paper are in
[`tutorials/paper_inputs`](https://github.com/TheoFEM-TUM/Hamster.jl/tree/main/tutorials/paper_inputs). If you use
Hamster or these data, please cite:

> M. Schwade *et al.*, Nat. Commun. **17**, 2652 (2026), [doi:10.1038/s41467-026-70865-7](https://doi.org/10.1038/s41467-026-70865-7)
