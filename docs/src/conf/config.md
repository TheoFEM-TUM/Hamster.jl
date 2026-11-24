# [Main Input File](@id man-config)

The main input file for a Hamster run is the `hconf` file. It defines the key settings for the run and determines which type of calculation is performed (standard run, optimization, or hyperparameter optimization).

The file is divided into a set of blocks, each of which has the following structure:

```bash
begin BLOCK_LABEL
    ...
end
```

The general [Options](@ref options-block) block is always required, while the presence of other blocks depends on the
type of calculation being performed. All input tags are listed in the [index of input tags](@ref block-labels) and
described on one page per block (in the navigation under Input Tags).

## Finding and checking tags

- `hamster --tag <name>` shows the description, block, type and default of a tag (all blocks in which it exists), and
  suggests similar names for typos.
- `hamster --tags <block or term>` lists the tags of a block (e.g. `hamster --tags ML`) or all tags whose name or
  description contains the term (e.g. `hamster --tags ewald`).
- When Hamster reads the config file, it warns about unknown tags (with suggestions) and about tags in a block where they
  have no effect, e.g. `xdatcar` in the `Options` block instead of the `Supercell` block.
- Values that do not fit the type of a tag (e.g. `rcut = 5,5`) or are not one of its allowed values (e.g. `loss = RMSE`;
  the allowed values are listed with the tag and are not case sensitive) stop the run with an error that lists all such
  tags, also for tags given on the command line.
- Lines that cannot be read stop the run with an error and the line number: a line in a block without `=` or with more
  than one `=`, a tag without a value, a block without `end` (or a `begin` inside a block) and an `end` without `begin`.
  Text outside the blocks is ignored.

Keys with an index, such as the band and k-point weights `wE_3` or `wE_10-20` of the `Optimizer` block, are described
with their base tag (`wE`, `wk`).

## Type of Calculation

The type of calculation is selected according to the configuration:

- If an [`Optimizer`](@ref optim-tags) block is present, a standard parameter optimization run is executed (see [here](@ref optim-run)).
- If a [`HyperOpt`](@ref hyperopt-tags) block is present, hyperparameter optimization is performed. This requires an accompanying [`Optimizer`](@ref optim-tags) block (see [here](@ref hyperopt-run)).
- If neither block is provided, a [standard run](@ref hyperopt-run) is performed.

## Structures from a trajectory

With a `Supercell` block (and `train_mode`/`val_mode = md` in fits), the structures are configurations of the trajectory
`xdatcar` (an XDATCAR or an HDF5 file), with `poscar` of the `Supercell` block as the reference structure. Configuration
`n` of the trajectory belongs to structure `n` of the HDF5 data files of a fit (`train_data`, `val_data`). The
configurations are given by `config_inds` (and `val_config_inds`), and `Nconf` configurations are sampled at random
between `Nconf_min` and `Nconf_max` (without those of `config_inds`). Before the run starts, all indices are checked:
the run stops with an error if an index is not in the trajectory or the data file, if `Nconf_max` is larger than the
number of configurations, or if fewer than `Nconf` configurations are available between `Nconf_min` and `Nconf_max`.
In universal mode, `Nconf` and `Nconf_max` are reduced to the number of configurations of every system, and HDF5
files of `config_inds` and `val_config_inds` can give the indices of every system in its group. The validation
structures of a fit can come from a trajectory of their own (`val_xdatcar`, see [Validation set](@ref)).

## Which positions the pair cutoff is applied to

A pair of ions enters the Hamiltonian only if it is close enough: with the default `rcut_pair_selection = reference`,
its distance in the **reference** structure (the `poscar` of the `Supercell` block in `md` mode, the first frame of
each system in universal mode) must be within `rcut`, while its distance in the current (distorted) frame only has
to be below `rcut + |rcut_tol|` and sets the smooth cutoff. With `rcut_pair_selection = frame`, a pair is included
iff its distance in the current frame lies in the support of the cutoff function: below `rcut + rcut_tol` for
`rcut_tol > 0`, below `rcut` for `rcut_tol <= 0`. The Hamiltonian then depends only on the current geometry, not on
the choice of the reference frame.

The same pair set is used for the TB model and the ML descriptors (including the environmental descriptor and the
`sepNN` shell cutoffs, which use the distorted distances in `frame` mode). With `frame` and an ML model,
`apply_distortion` and `apply_distance_distortion` of the `ML` block must be `true` (otherwise the run stops with an error), so that the descriptor uses the distorted distances and angles together with the frame pair set. In `frame` mode with `rcut_tol <= 0` the cutoff is hard at `rcut` on the frame distance, so pairs enter and leave the Hamiltonian when they cross `rcut` between frames and H can jump in MD; a warning is printed in this case, and a positive `rcut_tol` (smooth tail beyond `rcut`) is recommended.

In `frame` mode, the neighbor search must also cover the distorted distances: the grid cells (`grid_size`) and the
translation vectors are enlarged automatically to `rcut + max(rcut_tol, 0) + 2 dmax`, where `dmax` is the largest
displacement of an ion in any frame of the system (a warning is printed if an explicitly set `grid_size` is
enlarged; translation vectors passed by the caller are checked with a warning). In `pc` mode there are no
distortions, so the distorted positions are the reference positions.

The setting is stored in the model files and checked against the config file (like `rcut`), so a model fitted with
one setting is not silently evaluated with the other: `params.dat` headers are compared by `check_consistency`
(a missing key in files written before this tag existed counts as `reference`), `model.h5` files store the setting
(a warning is printed if the config file sets a different value; the stored value is used) and `ml_params.dat`
headers store it as well (a mismatch stops the run with an error).

With the default `reference`, a warning is printed once per run in `md`/`universal` runs if bonds are missed, i.e.
pairs beyond `rcut` in the reference structure whose distorted distance is below `rcut` (listed per element pair
over all structures). Consider `rcut_pair_selection = frame` if such bonds should have hoppings — in universal mode
the reference is only the first frame of each system, so `frame` is usually the appropriate choice there.

The default `reference` is kept for compatibility: existing parameter and model files reproduce their Hamiltonians
exactly.


## Type of model

The type of model is determined the following way:
- A TB model is always contructed.
- A $\Delta$-ML is added if an ML block is present.
- A SOC model is added if a SOC block is present (or `soc=true` in Options).