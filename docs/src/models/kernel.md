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

## Prediction uncertainty

If `uncertainty = true` (default) in the `ML` block, the model summary also reports the *prediction uncertainty*: the
Gaussian-process posterior variance of the kernel regression at each descriptor,
``\mathrm{var}(x) = 1 - k(x, D)\,(K + \lambda I)^-1 k(D, x)``, with kernel width ``\sigma`` = `sim_params` and noise
``\lambda`` = `unc_noise` (default 1e-3). It is computed in a single pass from the data points (it does not depend on the
fitted parameters): about 0 at a data point, close to 1 far from all data points. The table gives the mean, 99th
percentile and maximum variance per group. The full per-structure score is also available programmatically as
`get_uncertainty(ham, i)`.

The per-structure *out-of-distribution score* printed below the table is cheaper: the maximum over the atoms of the
per-atom mean of ``1 - \max_k \mathrm{sim}(x, d_k)^2`` over **all** descriptors (an upper bound of the GP variance,
computable in O(N) per descriptor together with the coverage), so that a local out-of-distribution environment (e.g. a
defect in a large cell) is not diluted by averaging over all atoms.

A structure is flagged as out of distribution (warning in `hamster.out` and in the log) if its score exceeds
`max(unc_threshold, unc_rel * unc_ref)`: the absolute `unc_threshold` (default 0.5), raised to `unc_rel` (default 2)
times the training reference `unc_ref` — the median score of the training structures, stored in the header of
`ml_params.dat` at fit time (fallback if unavailable: the median over the structures of the current run). Note that
the absolute scale of the scores depends on the number of data points and `sim_params`: with few, narrowly separated
data points, even well-covered descriptors have a large variance.

The cost of the GP variance is one triangular solve with the ``N`` data points per descriptor, ``O(N^2)``, so the
summary table estimates it from at most `unc_max_desc` (default 10000) randomly chosen descriptors per structure (with a
fixed seed, independent of the MPI decomposition). The full score for individual structures can be computed on demand
with `get_uncertainty`.

The same variance is used for sampling data points with `sampling = uncertainty` in the `ML` block: within each
k-means cluster, the descriptors with the largest posterior variance with respect to the already selected points are
chosen greedily (seeded with the cluster centroids), i.e. the most out-of-distribution descriptors of each cluster.
This tends to select data points that improve the fit more than random sampling, at a higher one-time cost during the
fit (the candidate variances are updated incrementally after each pick).



## Parameter file

A fit writes the ML parameters and data points to `ml_params.dat` (tag `filename` of the `ML` block). Its header lists
the settings that define the model: `rcut`, `sim_params`, `env_scale`, `strc_scale`, `apply_distortion`,
`apply_distance_distortion`, `apply_orthogonality` and `key_dims` (if set). When the file is loaded (`init_params` of
the `ML` block, or `filename` with `mode = expand`), settings that are not given in the `ML` block are taken from the
header, so the model is used with the settings it was fitted with. Settings that are given and differ raise an error.
Files of earlier versions, whose header has only some of these settings, use the defaults for the others.
