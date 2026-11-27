# Kernel-based ML corrections

## Model Summary

When the model is built in a Hamster run, a summary is appended to `hamster.out`, with the number of descriptor
vectors (one per matrix element of all structures) and the coverage of the descriptors by the data points of the
kernel:

```
================================================================================
  ML Model Summary (validation set)
--------------------------------------------------------------------------------
      descriptor vectors:          20100

  Coverage: similarity of every descriptor to its most similar data point
    group        descriptors    min     p1     p5 median  <tol %  <0.5 %
    Br onsite           1296  0.744  0.776  0.879  0.984    0.00    0.00
    Pb onsite            768  0.151  0.335  0.688  0.946    0.00    2.21
    Br+Br              11124  0.041  0.157  0.713  0.954    0.00    3.74
    Br+Pb               6912  0.269  0.520  0.693  0.920    0.00    0.87

    Similarity exp(-|x - d|² / 2σ²) with σ = sim_params; percentiles with a
      resolution of 0.001.
    <tol %: fraction of descriptors with a similarity below sim_tol; <0.5 %:
      below 0.5.
    Descriptors with a small similarity to all data points are outside the
      training data; there, the ML correction is close to zero and the model
      falls back to TB.
================================================================================
```

For every descriptor ``x``, the coverage is the similarity to its most similar data point,
``\max_k \exp(-|x - d_k|^2 / 2\sigma^2)`` with ``\sigma`` = `sim_params`, grouped by ion pair and onsite terms.
Descriptors with a small similarity to all data points lie outside the training data: there, the ML correction is close
to zero and the model falls back to TB. `<tol %` and `<0.5 %` give the fraction of descriptors with a similarity
below `sim_tol` and 0.5. The validation
and evaluation models use the data points of the training (or loaded) model, so their coverage shows how well the
training data cover new structures, e.g. larger cells.


## Precomputed similarities (`sim_mat`)

With `sim_mat = true` in the `ML` block, Hamster computes the similarities between the descriptor of every matrix
element and the kernel points once, when the model is built, and keeps those above `sim_tol`. With `key_dims`, only the
kernel points with the same values of these (discrete) descriptor dimensions are compared, e.g. the same pair of
species. Building a Hamiltonian is then a sparse product of the stored similarities with the parameters, which saves
the kernel evaluation in every iteration of a fit.

The stored similarities take about 12 bytes each (one `Float64` value and one `Int32` index, packed per lattice
vector). How many are kept depends on the number of kernel points and their width `sim_params`: with thousands of kernel
points, every matrix element can keep hundreds of them, and the precomputed matrix then needs far more memory than the
descriptors themselves (64 bytes per matrix element). For a single evaluation, e.g. a standard run of large cells, the
direct evaluation (`sim_mat = false`) is therefore usually faster and needs less memory. The direct evaluation is also
exact, while the precomputed matrix drops the similarities below `sim_tol`.

`sim_mat`, `sim_tol` and `key_dims` only determine how the model is evaluated. They are not stored in model files
(`save_model`); when a model is loaded, the values of the config file apply.

## Parameter file

A fit writes the ML parameters and data points to `ml_params.dat` (tag `filename` of the `ML` block). Its header lists
the settings that define the model: `rcut`, `sim_params`, `env_scale`, `strc_scale`, `apply_distortion`,
`apply_distance_distortion`, `apply_orthogonality` and `key_dims` (if set). When the file is loaded (`init_params` of
the `ML` block, or `filename` with `mode = expand`), settings that are not given in the `ML` block are taken from the
header, so the model is used with the settings it was fitted with. Settings that are given and differ raise an error.
Files of earlier versions, whose header has only some of these settings, use the defaults for the others.
