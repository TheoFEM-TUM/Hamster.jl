# [TB Hyperparameter optimization for GaAs](@id hyperopt-gaas)

This example demonstrates hyperparameter optimization of a TB model for GaAs using a hybrid orbital basis. While all numerical input parameters can in principle be optimized, we focus here on the distance-dependence parameter `alpha`. All input files can be found [here](https://github.com/TheoFEM-TUM/Hamster.jl/tree/main/examples/gaas_tb_hyperopt).

To run a hyperparameter optimization, both an `Optimizer` and a `HyperOpt` block must be present in the config file. At minimum, the parameters to be optimized and their respective search windows (`lowerbounds` and `upperbounds`) need to be specified. The search interval can further be refined using the `stepsizes` argument.

Three hyperparameter optimization methods are available: Random Search (`method=random`), Grid Search (`method=grid`), and the Tree-structured Parzen Estimator (`method=tpe`). The desired method can be selected via the `method` argument.

As additional input files, two data sets (training and validation) as well as a structure file are required.

By default, this workflow reports an update only after each completed hyperparameter optimization step, showing the final validation loss and the current optimum for comparison. An estimate of the remaining runtime is also provided, based on the average iteration time.