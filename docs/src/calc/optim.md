# [Parameter Optimization](@id optim-run)

```@docs
Hamster.run_calculation(::Val{:optimization}, ::Any, ::Config)
```

## Weights of the bands and k-points

By default, every eigenvalue counts the same in the loss (the mean of |ε - ε_ref|ⁿ over bands and k-points, n = 1 for
`loss = MAE`, 2 for `MSE`). There are two ways to weight them:

- `wE` and `wk` (and `wE_<band>`, `wk_<k>` for single bands or ranges) in the `Optimizer` block weight the bands and
  k-points by their index. They need the same number of bands and k-points in all training structures and are ignored
  otherwise, e.g. in universal mode with systems of different size.
- A dataset `weights` in the HDF5 eigenvalue file, next to `eigenvalues`: one weight per eigenvalue. It works in all
  modes (pc, md, universal: in the group of each system) and for any number of bands per system, so it can, for
  example, put more weight on the band edges or on a defect level of each structure, or exclude bands with weight 0.

The `weights` dataset has the shape of the eigenvalues (bands × k-points × structures), bands × k-points (the same for
all structures) or bands (the same for all k-points and structures). It covers the bands of the file, like
`eigenvalues`; the fitted bands (`bandmin`, `bandinclude`) are taken from it in the same way. The weights must be
finite and non-negative and not all zero for the fitted bands of a structure.

Every error |ε - ε_ref|ⁿ is multiplied by its weight, with the same normalization as without weights: weights of one
give the unweighted loss, and a weight of 2 on all eigenvalues of a system doubles its share of the total loss. With
`wE` and `wk` as well, the weights are multiplied. The validation loss uses the weights of the validation data only if
`val_weights = true`.

```julia
using HDF5
h5open("eigenval.h5", "r+") do f
    E = read(f["eigenvalues"])                  # bands × k-points × structures
    W = ones(size(E))
    W[end-3:end, :, :] .= 0                       # e.g. bands above the converged DFT bands
    f["weights"] = W
end
```
