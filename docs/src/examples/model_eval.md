# [Standard run for GaAs](@id standard-gaas)

If no optimization blocks are defined, Hamster defaults to a standard run, computing the Hamiltonian and eigenvalues. In this case, an optimized parameter set must be provided via the `init_params` tag. The distance-dependence functions can either be supplied directly or re-interpolated as needed. Note that the cutoff `rcut` and the distance-dependence parameters `alpha` must remain consistent across calculations; otherwise, the results may be erroneous. Input files are found [here](https://github.com/TheoFEM-TUM/Hamster.jl/tree/main/examples/gaas_model_eval).

For each material to be modeled, an orbital basis must be specified. Hybrid orbitals are a special case since they are identical apart from their spatial orientation. The `nnaxes` argument can be used to automatically set these orientations based on the bonding axes between an atom and its nearest neighbors (e.g., As in this case).

By default, the Hamiltonian will be diagonalized to compute the energy eigenvalue spectrum for the provided k-points and store them in a file `Es.dat`. This file can be read using the `read_from_file` function from the Hamster package.
## Exporting and reloading a model

Set the following Options tags in a calculation or fitting configuration.

```text
begin Options
    save_model = true
    model_file = model.h5
end
```

A fit exports its final fitted weights. The HDF5 file contains TB parameters, active
SOC and Ewald parameters, ML weights and support points, atom and model settings,
and radial interpolation tables. Geometry tensors and ML similarity matrices are
rebuilt for the structures requested by the receiving calculation.

For evaluation in another directory, copy `model.h5` and provide the target geometry.

```text
begin Options
    load_model = true
    model_file = model.h5
    poscar = POSCAR
end
```

Stored atom and model settings take precedence over the corresponding input tags.
The receiving configuration supplies calculation inputs such as geometry, k-points,
and output settings. The model file supplies the fitted parameters and radial tables.
Every requested TB, SOC and Ewald parameter label must be present in the file.

The Julia API uses the same workflow.

```julia
save_model(ham, strcs, bases, conf, comm; filename="model.h5", rank=rank)
load_model!(conf, comm; filename="model.h5", rank=rank)
strc = Structure(conf)
basis = Basis(strc, conf)
ham = EffectiveHamiltonian([strc], [basis], comm, conf; rank=rank, nranks=nranks)
```

All ranks of `comm` participate in export and loading. Rank zero writes a temporary
file and replaces the destination after a successful write. Loading a missing,
incomplete or unsupported-version file raises an error. The format version is stored
in the `version` dataset.

### Model file API

```@docs
Hamster.save_model
Hamster.load_model!
```
