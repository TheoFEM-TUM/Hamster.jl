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

