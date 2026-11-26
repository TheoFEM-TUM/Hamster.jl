# [Parameter Optimization](@id optim-run)

```@docs
Hamster.run_calculation(::Val{:optimization}, ::Any, ::Config)
```

## Validation set

A fit is validated if `val_data`, `val_mode` or `val_ratio` is set. With `val_mode = pc`, the validation set is the
primitive cell of `val_data` (also when the training data are MD structures). With `val_mode = md` (or universal), it
consists of round(`Nconf` × `val_ratio`) configurations between `Nconf_min` and `Nconf_max` that are not in the
training set, or of `val_config_inds`. If this leaves no validation structure, the fit stops with an error.

The validation structures can also come from a trajectory of their own, e.g. another temperature or larger cells:
`val_xdatcar` (and `val_poscar`, its reference structure in md mode) in the `Supercell` block, with the eigenvalues in
`val_data` (structure `n` of `val_data` belongs to configuration `n` of `val_xdatcar`). The validation set is then all
configurations of this file, or those of `val_config_inds`, independent of the training set; `Nconf`, `val_ratio`,
`Nconf_min` and `Nconf_max` apply only to the training set. Prepared training and validation files (e.g. subsets of a
larger data set) are used completely with `Nconf` = `Nconf_max` = number of training configurations and `val_xdatcar`.
Earlier versions read `val_xdatcar` and `val_poscar` from the `Optimizer` block; config files with them there still
work, with a warning.

```
begin Supercell
    xdatcar = train/structures.h5
    val_xdatcar = val/structures.h5
end

begin Optimizer
    train_data = train/eigenval.h5
    val_data = val/eigenval.h5
    val_mode = md
end
```

In universal mode, `val_xdatcar` has one group per system, like `xdatcar`; it may contain only some of the systems of
the training set, or other systems whose species are in the training set. The indices of `config_inds` and
`val_config_inds` given as an HDF5 file are read from the group of each system (`<system>/train_config_inds`,
`<system>/val_config_inds`, the layout of `hamster_out.h5` of a universal fit), so every system can have its own
configurations; datasets at the top of the file apply to all systems.

Without `val_xdatcar`, the validation structures come from `xdatcar`, whose systems are read with `train_mode`: universal
validation structures then need `train_mode = universal`. To validate a fit of other training data (e.g. a primitive
cell, `train_mode = pc`) on a universal file, give the file as `val_xdatcar`; a `Supercell` block that only sets
`val_xdatcar` (and `val_poscar`) does not change the default `train_mode` (pc).

```
begin Supercell
    val_xdatcar = structures.h5
end

begin Optimizer
    train_mode = pc
    train_data = EIGENVAL
    val_mode = universal
    val_data = eigenval.h5
end
```

Before the structures are read, the trajectories and HDF5 data files are checked against the modes that read them: md
mode reads a single trajectory (an XDATCAR, or an HDF5 file with `positions` and `lattice` at the top level, data with
`eigenvalues` at the top level), universal mode an HDF5 file with one group per system (`positions`, `lattice`,
`atom_types`; data with `eigenvalues` in the group of every system). A file of the other layout stops the run with an
error that names the input tag, the file and the mode.

With MPI, the training structures are split over the ranks and the validation structures over the ranks with training
structures, so a rank can have training structures but no validation structure. The validation loss is the mean over
all validation structures.

## Batches

With `nbatch` > 1 (`Supercell` block), the parameters are updated `nbatch` times per iteration, each time with the
gradient of one batch of training structures. With MPI, every rank splits its own training structures into `nbatch`
batches, and batch `b` of the fit consists of batch `b` of all ranks. A rank with fewer structures than `nbatch` has
empty batches at the end. If `nbatch` is larger than the number of training structures of every rank, the fit uses as
many batches as the rank with the most structures has (with a warning).

## Weights of the bands and k-points

By default, every eigenvalue counts the same in the loss (the mean of |ε - ε_ref|ⁿ over bands and k-points, n = 1 for
`loss = MAE`, 2 for `MSE`). There are two ways to weight them:

- `wE` and `wk` (and `wE_<band>`, `wk_<k>` for single bands or ranges) in the `Optimizer` block weight the bands and
  k-points by their index. They need the same number of bands and k-points in all training structures and are ignored
  otherwise, e.g. in universal mode with systems of different size.
- A dataset `weights` in the HDF5 eigenvalue file, next to `eigenvalues`: one weight per eigenvalue. It works in all
  modes (pc, md, universal: in the group of each system) and for any number of bands per system, so it can, for
  example, put more weight on the band edges or on a defect level of each structure, or exclude bands with weight 0.

The `weights` dataset has the shape of the eigenvalues (bands × k-points × structures), bands × k-points (the same for
all structures) or bands (the same for all k-points and structures). It covers the bands of the file, like
`eigenvalues`; the fitted bands (`bandmin`, `bandinclude`) are taken from it in the same way. The weights must be
finite and non-negative and not all zero for the fitted bands of a structure.

Every error |ε - ε_ref|ⁿ is multiplied by its weight, with the same normalization as without weights: weights of one
give the unweighted loss, and a weight of 2 on all eigenvalues of a system doubles its share of the total loss. With
`wE` and `wk` as well, the weights are multiplied. The validation loss uses the weights of the validation data only if
`val_weights = true`.

```julia
using HDF5
h5open("eigenval.h5", "r+") do f
    E = read(f["eigenvalues"])                  # bands × k-points × structures
    W = ones(size(E))
    W[end-3:end, :, :] .= 0                       # e.g. bands above the converged DFT bands
    f["weights"] = W
end
```
