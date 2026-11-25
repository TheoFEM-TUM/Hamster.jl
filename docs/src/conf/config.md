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

`rcut_pair_selection` decides which pairs of ions get matrix elements:

- `reference` (default): the pair must be within `rcut` in the **reference** structure (the `poscar` of the
  `Supercell` block in `md` mode, the first frame of each system in universal mode). The current distance only sets
  the smooth cutoff. A system should then be one trajectory close to its reference; frames from other runs or phases
  get a wrong set of bonds.
- `frame`: the pair must be within the cutoff in the **current** frame (below `rcut + rcut_tol`, or below `rcut` for
  `rcut_tol <= 0`). The Hamiltonian no longer depends on the choice of the reference. Recommended for universal data
  sets and for frames far from their reference.

Notes on `frame`:
- TB model and ML descriptors use the same pairs. With an ML model, `apply_distortion` and
  `apply_distance_distortion` must be `true`.
- The `sepNN` shell cutoffs still come from the reference positions, so they are fixed per system.
- Keep `rcut_tol > 0`: with `rcut_tol <= 0` the cutoff is hard and H jumps when a pair crosses `rcut` (warning).
- The neighbor search grows to `rcut + max(rcut_tol, 0) + 2 dmax` (`dmax`: largest displacement from the
  reference), which makes the setup slower for frames far from the reference.

The setting is stored in `params.dat`, `ml_params.dat` and `model.h5`. A `params.dat` or `ml_params.dat` that
differs from the config file (files without the setting count as `reference`) gives a warning or an error;
`model.h5` uses its stored value. In `reference` mode, a warning lists bonds that are within `rcut` in a frame but
missing because they are beyond `rcut` in the reference.

## Cutoffs per pair of elements

`rcut_<A>_<B>` in the Options block sets the cutoff of the pairs of elements `A` and `B` (the order does not matter,
`rcut_Br_Pb` = `rcut_Pb_Br`); all other pairs use `rcut`:

```
begin Options
    rcut       = 6.5
    rcut_Br_Pb = 4.5     # first Pb-Br shell only
    rcut_Br_Br = 5.5
    rcut_Br_Cs = 6.0
    rcut_Pb_Pb = 0       # no Pb-Pb hopping
    rcut_Cs_Cs = 0
end
```

- A pair cutoff lies between 0 and `rcut`; `rcut` sets the neighbor search and is the largest cutoff.
- The pair cutoff replaces `rcut` in the pair selection (`rcut_pair_selection`) and in the smooth cutoff
  (`rcut_tol`) of the TB model and of the ML descriptors (there at most the `rcut` of the ML block).
- `0` removes all matrix elements between different ions of the pair; the on-site terms stay, without a smooth
  cutoff. The TB parameters of these pairs are set to zero (also when they come from `init_params`) and stay zero
  in a fit. With `onsite = false`, the on-site terms of a species use its nearest-neighbor parameters, which are
  kept.
- In distorted structures, a cutoff between two neighbor shells of a pair (where no distances occur in any of the
  structures) makes the set of bonds of this pair the same for the reference and the frames. Pairs with distances
  around their cutoff need the smooth cutoff (`rcut_tol > 0`) and `rcut_pair_selection = frame`.

The pair cutoffs are written to `params.dat`, `ml_params.dat` and `model.h5`. A `params.dat` with other pair cutoffs
gives a warning, an `ml_params.dat` an error (a file without them has `rcut` for all pairs), and `model.h5` uses its
stored values. Pair cutoffs can be hyperparameters (`params = rcut_Br_Pb` in the `HyperOpt` block).


## Type of model

The type of model is determined the following way:
- A TB model is always contructed.
- A $\Delta$-ML is added if an ML block is present.
- A SOC model is added if a SOC block is present (or `soc=true` in Options).