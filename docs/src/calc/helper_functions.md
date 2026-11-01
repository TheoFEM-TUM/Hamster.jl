# Helper Functions

```@docs
Hamster.get_eigenvalues
```

```@docs
Hamster.hyper_optimize
```

## Setup of the calculations

The standard run and the optimization share these steps: the structures are split over the MPI ranks, the ranks
with data get their own communicator, and the structures, bases and effective Hamiltonian are built.

```@docs
Hamster.local_structure_indices
Hamster.active_communicator
Hamster.build_hamiltonian
```
