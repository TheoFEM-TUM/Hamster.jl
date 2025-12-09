# [Main Input File](@id man-config)

The main input file for a Hamster run is the `hconf` file. It defines the key settings for the run and determines which type of calculation is performed (standard run, optimization, or hyperparameter optimization).

The file is divided into a set of blocks, each of which has the following structure:

```bash
begin BLOCK_LABEL
    ...
end
```

Tags placed outside their appropriate blocks are ignored. The general [Options](@ref options-block) block is always required, while the presence of [other blocks](@ref block-labels) depends on the type of calculation being performed.

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