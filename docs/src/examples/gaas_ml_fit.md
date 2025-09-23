# [Fitting a Delta-ML model on data from MD](@id mdfit-gaas)

This example demonstrates how to fit a Δ-ML model on top of an existing TB parameterization for GaAs. While all Hamiltonian components (TB, ML, SOC) can in principle be optimized simultaneously, it is significantly more stable to fit them separately.

To this end, we will use eigenvalue data (provided by `eigenval.h5`) from an MD trajectory (provided by `structures.h5`) computed with VASP. A `POSCAR` file (e.g., initial structure of the MD run) is also required to provide atom species information and reference positions. Inputs file can be found [here](https://github.com/TheoFEM-TUM/Hamster.jl/tree/main/examples/gaas_ml_fit).

## Supercell block

The `supercell` block tells `Hamster` to analyze multiple structures.  
The keywords `poscar` and `xdatcar` specify the reference `POSCAR` file and the trajectory file (`XDATCAR` or `*.h5`).  

The selection of structures is controlled by:  
- `nconf`: Number of structures to sample.  
- `nconf_min`: Minimum trajectory index (optional, default = 1).  
- `nconf_max`: Maximum trajectory index (required).

## ML block

The ML model uses a Gaussian kernel to interpolate between a set of descriptor vectors of Hamiltonian matrix elements.  
In practice, it is recommended to perform hyperparameter optimization for the ML model.  

**Sampling of descriptor vectors** is controlled by:  
- `npoints`: Number of descriptor vectors to sample.  
- `ncluster`: Number of clusters used in k-means clustering.  

**Kernel model parameters** can be modified via:  
- `sim_params`: Kernel width.  
- `env_scale`: Scaling factor applied only to the environment value.  
- `apply_distortion`: If enabled, all values in the descriptor vector are updated according to distorted atomic positions.


## Optimization

Both `train_mode` and `val_mode` should be set to `md`. The optimization can be parallelized over the number of structures using MPI. Set `update_tb` to `false` so that only the ML parameters are updated. Using a small learning rate (e.g., 0.01) is recommended to improve optimization stability.