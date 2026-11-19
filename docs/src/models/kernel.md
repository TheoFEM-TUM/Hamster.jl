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
``\\mathrm{var}(x) = 1 - k(x, D)\\,(K + \\lambda I)^{-1} k(D, x)``, with kernel width ``\\sigma`` = `sim_params` and noise
``\\lambda`` = `unc_noise` (default 1e-3). It is computed in a single pass from the data points (it does not depend on the
fitted parameters): about 0 at a data point, close to 1 far from all data points. The table gives the mean, 99th
percentile and maximum variance per group, and the per-structure score is the per-atom mean of the per-element
variances, averaged over the atoms (between 0 and 1), also available programmatically as `get_uncertainty(ham, i)`.

A structure is flagged as out of distribution (warning in `hamster.out` and in the log) if its mean uncertainty exceeds
`unc_threshold` (default 0.5); with at least two structures in the run, the threshold is
`max(unc_threshold, unc_rel * median)`, i.e. relative to the other structures of the run (the training structures in a
fit), so that the warning also fires for sparse models whose absolute variances are high everywhere. Note that the
absolute scale of the variance depends on the number of data points and `sim_params`: with few, narrowly separated
data points, even well-covered descriptors have a large variance.

The cost of the variance is one triangular solve with the ``N`` data points per descriptor, ``O(N^2)``, so the summary
estimates it from at most `unc_max_desc` (default 10000) randomly chosen descriptors per structure. The full score for
individual structures can be computed on demand with `get_uncertainty`.

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
