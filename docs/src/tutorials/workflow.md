# [Building a model](@id tutorial-workflow)

This page summarizes how a Hamster model is built: the building blocks, the order in which they are fitted, and our
recommendations. The tutorials go through every step with an example.

## Building blocks

A Hamster Hamiltonian is a sum of terms, each switched on by a block in the config file:

| Term | Block | What it describes | Fitted to | Page |
|---|---|---|---|---|
| Tight binding (TB) | always (`Options`, `atom` blocks) | the electronic structure of the ideal (or average) structure, with Slater-Koster parameters that depend on the distance | primitive cell band structure | [TB model](../models/tb.md) |
| Spin-orbit coupling | `SOC` | on-site SOC, one parameter per species | band structure with SOC | [SOC model](../models/soc.md) |
| ML corrections | `ML` | corrections of every matrix element to changes of the atomic environment (Δ-ML, Gaussian kernel) | MD snapshots | [Kernel model](../models/kernel.md) |
| Electrostatics | `Ewald` | on-site shifts by the electrostatic potential of the ions (Ewald, PME, Wolf, Zahn) | MD snapshots | [Ewald model](../models/ewald.md) |

The basis is set per species in the `atom` blocks: the orbitals (`orbitals`, e.g. `s px py pz` or four `sp3dr2` hybrids)
and the radial functions (Slater-type, principal quantum number `n` and effective charge `alpha`), which set how the
matrix elements change with distance and direction.

## Steps

1. **DFT data** ([tutorial 1](@ref tutorial-data)): a band structure of the primitive cell (training and validation
   k-points), with SOC if needed, and eigenvalues of snapshots of an MD run (supercells) for the ML and
   electrostatics terms.
2. **Basis**: choose the orbitals of every species that make up the bands of interest, typically the bands around the
   gap. Fewer orbitals give faster models; the bands far from the gap are then described less well.
3. **TB fit** of the primitive cell ([tutorial 3](@ref tutorial-tb-fit)), with hyperparameter optimization of the
   radial functions ([tutorial 4](@ref tutorial-hyperopt)). Check the band structure at k-points not used in the
   fit.
4. **SOC** ([tutorial 5](@ref tutorial-soc)): fit the SOC parameters on top of the fixed TB parameters.
5. **ML corrections** ([tutorial 6](@ref tutorial-ml)) for distorted structures, trained on MD snapshots with the TB
   (and SOC) parameters fixed. Optimize the hyperparameters of the kernel (`npoints`, `ncluster`, `sim_params`).
6. **Electrostatics** ([tutorial 7](@ref tutorial-electrostatics)), for ionic materials, before or together with the
   ML corrections.
7. **Validation and application** ([tutorial 8](@ref tutorial-applications)): test the model on snapshots not used
   in the fit (e.g., another temperature), then use it for large cells, DOS, PDOS, band edges and more.

All terms can in principle be fitted at the same time, but fitting them one after the other (with the parameters of
the previous steps fixed via `update_tb = false`) is much more stable.

## Recommendations

**Data**

- Use the same DFT settings (functional, PAW potentials, SOC) for all data of one model, so that the eigenvalues have
  the same reference.
- Put a dense set of k-points near the band edges into the training data, and give the band edges a higher weight
  (`wk`, `wE`) if the gap matters for your application.
- For the ML term, sample snapshots from MD runs at the temperatures you want to describe; a few tens of snapshots of
  small supercells are often enough, since every snapshot contains many atomic environments.

**TB fit**

- `rcut` must include the neighbor shells that couple the orbitals of the bands of interest (in a cubic perovskite,
  e.g., the Pb–Br and Br–Br neighbors). Larger values give more parameters and slower models.
- Start with a learning rate of 0.1–0.3 for TB fits, and 0.01 or less for ML fits; reduce it if the loss jumps.
- Exclude bands that the basis cannot describe with `bandmin` (bands below) and the number of orbitals (bands above),
  and down-weight the highest model bands with `wE`.
- Always validate: on other k-points (primitive cell) or other snapshots (`val_ratio`, `val_data`).

**ML corrections**

- Keep the TB parameters fixed (`update_tb = false`).
- `npoints` (kernel points) controls the accuracy and the cost; increase it until the validation loss no longer
  improves. The ML summary in `hamster.out` shows how well the kernel points cover the descriptors of the data.
- Descriptors far away from all kernel points get no correction: the model falls back to TB there.

**Electrostatics**

- Use effective charges that make every cell neutral (`qeff`), and short-ranged methods (`wolf`, `zahn`) for models
  that are applied to cells much larger than the training cells.
- Keep a term only if it improves the validation loss.

## Troubleshooting

| Problem | Possible cause and fix |
|---|---|
| Warning "Unknown input tag" | typo or wrong block: see the suggestion in the warning and `hamster --tag <name>` |
| Bounds error when reading eigenvalues | `bandmin` + number of model bands exceeds the bands in the data, or `Nconf_max` exceeds the snapshots in the file |
| Loss does not decrease or jumps | learning rate too large (`lr`), or wrong `bandmin` (model bands compared with the wrong DFT bands) |
| Large loss from the start | wrong `bandmin`: count the model bands below the VBM (occupied orbitals per cell × cells) |
| Training loss much smaller than validation loss | too many parameters for the data (`rcut`, `npoints`) or too few training structures/k-points |
| SOC has no effect | missing `SOC` block in the run, or SOC parameters not read (`init_params` of the `SOC` block) |
| Slow fits | use `sp_mode = true`, threads (`-t`), MPI over structures, or a GPU (see [GPU support](@ref gpu-support)) |
