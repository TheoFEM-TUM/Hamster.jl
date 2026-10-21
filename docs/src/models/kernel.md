# Kernel-based ML corrections

## Symmetry of the Corrections

The ML correction of a matrix element is a scalar kernel value of its descriptor,
``\Delta H_{ij} = \sum_k w_k \exp(-|x_{ij} - d_k|^2 / 2\sigma^2)``. A matrix element is linear in both orbitals; the
kernel is not. A symmetry operation that maps an orbital onto its negative (a p orbital under inversion) or onto a
combination of orbitals (dz2 under a threefold axis, px under a hexagonal sixfold axis) therefore relates the
corrections of different elements in a way the kernel does not reproduce: without further measures, the corrections
break the symmetry of the structure (e.g. split Kramers pairs at generic k in a centrosymmetric crystal; at Γ and the
other time-reversal invariant points time reversal alone keeps them degenerate).

With `enforce_symmetry_group = O_h` or `D_6h` in the `ML` block, the corrections transform exactly like the Hamiltonian
under every operation of the point group (axes along the Cartesian axes; for `D_6h` the sixfold axis along z). The
kernel is averaged over the orbit ``P`` of the orbitals of each shell under the group (and the sign),

```math
\Delta H_{ij} = \sum_{u \in P_i} \sum_{v \in P_j} (S_i^{-1} u)\cdot c_i \, (S_j^{-1} v)\cdot c_j \; K(x(u, v)),
\qquad S = \sum_{u \in P} u u^\mathsf{T},
```

where ``c_i`` are the coefficients of orbital ``i`` in its shell and ``x(u, v)`` is the descriptor of the orbitals ``u``
and ``v`` (angles from their angular functions, squared environment ``u^\mathsf{T} E u``). This projects the kernel onto
functions linear in both orbitals; on the same site and shell, the pairs ``(u, u)`` are added so that isotropic shifts
remain. It holds for every structure (the model is continuous in the geometry and needs no symmetry detection), so
structures whose symmetry is a subgroup of the chosen group are exactly symmetric. For p and t2g orbitals under `O_h`
the orbits are the orbitals with both signs (2 variants per orbital); eg orbitals under `O_h` and in-plane p orbitals
under `D_6h` have larger orbits, which makes the ML part more expensive for these elements. Hybrid orbitals are not
averaged, and s orbitals are their own orbit (no sign average, which would remove their corrections). The shells must
be complete (e.g. all three p orbitals).

The corrections inherit a restriction of the descriptor: it describes an orbital by its angular function along the bond
and by sign-invariant environment values, so a p orbital exactly perpendicular to the bond gets no s-p correction, even
in an asymmetric environment that would allow such a coupling (as in the two-centre Slater-Koster form).

The group only helps if the structure is aligned with its axes. Hamster checks this for the lattice of every system:
`hamster.out` lists how many operations of the group map the lattice onto itself, compared with the point group of
the lattice; a warning is given if the lattice is more symmetric than the group finds (the structure is probably
rotated) or if the other group fits better.

The environment descriptor is the sum of squares ``e_i = \sum_k H_{ik}^2 f_\mathrm{cut}`` (sign-invariant). Its values
and especially their variation are much smaller than those of the linear environment, so `env_scale` usually has to be
increased for the kernel to resolve them (their spread should be comparable to `sim_params`). With `enforce_symmetry_group = none` (default), the
corrections are computed as before. The tag is written to the ML parameter file and taken from there as the default;
a model trained with a group cannot be evaluated without it.

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

