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
    group        descriptors     min      p1      p5  median <sim_tol %  <0.5 %
    Br onsite          1296   0.487   0.727   0.843   0.981      0.00    0.08
    Pb onsite           768   0.372   0.644   0.712   0.934      0.00    0.13
    Br+Br             11124   0.006   0.162   0.756   0.955      0.07    2.54
    Br+Pb              6912   0.195   0.455   0.675   0.932      0.00    1.48
================================================================================
```

For every descriptor ``x``, the coverage is the similarity to its most similar data point,
``\max_k \exp(-|x - d_k|^2 / 2\sigma^2)`` with ``\sigma`` = `sim_params`, grouped by ion pair and onsite terms.
Descriptors with a small similarity to all data points lie outside the training data: there, the ML correction is close
to zero and the model falls back to TB. `<sim_tol %` and `<0.5 %` give the fraction of such descriptors. The validation
and evaluation models use the data points of the training (or loaded) model, so their coverage shows how well the
training data cover new structures, e.g. larger cells.

