# Helper Functions

```@docs
Hamster.get_eigenvalues
```

## Hyperparameter optimization

```@docs
Hamster.hyper_optimize
Hamster.hyperopt_tag
Hamster.check_hyperopt_settings
Hamster.split_depends_on
Hamster.write_hyperopt_results
```

## Setup of the calculations

The standard run and the optimization share these steps: the structures are split over the MPI ranks, the ranks
with data get their own communicator, and the structures, bases and effective Hamiltonian are built.

```@docs
Hamster.local_structure_indices
Hamster.active_communicator
Hamster.build_hamiltonian
```
