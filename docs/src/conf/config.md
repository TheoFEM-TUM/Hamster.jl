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

Keys with an index, such as the band and k-point weights `wE_3` or `wE_10-20` of the `Optimizer` block, are described
with their base tag (`wE`, `wk`).

## Type of Calculation

The type of calculation is selected according to the configuration:

- If an [`Optimizer`](@ref optim-tags) block is present, a standard parameter optimization run is executed (see [here](@ref optim-run)).
- If a [`HyperOpt`](@ref hyperopt-tags) block is present, hyperparameter optimization is performed. This requires an accompanying [`Optimizer`](@ref optim-tags) block (see [here](@ref hyperopt-run)).
- If neither block is provided, a [standard run](@ref hyperopt-run) is performed.


## Type of model

The type of model is determined the following way:
- A TB model is always contructed.
- A $\Delta$-ML is added if an ML block is present.
- A SOC model is added if a SOC block is present (or `soc=true` in Options).