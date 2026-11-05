# Ewald on-site term for slabs and clusters

Sample configuration files for the electrostatic on-site term of systems with vacuum: a CsPbBr₃ slab (`slab/`) and a
CsPbBr₃ quantum dot (`quantum_dot/`). The data files they name (`structures.h5`, `XDATCAR`, `eigenval.h5`, `POSCAR`,
start parameters) are not included; replace the paths with your own. The background is on the Ewald page of the
manual (sections *Slabs* and *Clusters*).

## Slab (`slab/`)

| File | Content |
|---|---|
| `hconf_fit_wolf` | TB fit with Wolf and the plane-averaged long range of the slab (`slab_axis = 3`) |
| `hconf_fit_ewald` | the same with the full Ewald sum (particle mesh) and the dipole correction of the slab |
| `hconf_eval_wolf` | evaluation of the fitted model on all frames |

- `slab_axis` is the lattice vector across the vacuum (1, 2 or 3). The slab may cross the periodic boundary.
- With effective charges, the layers of a slab are usually charged (e.g. PbBr₂ and CsBr layers of CsPbBr₃ (001)).
  `charge_compensation = surface` (default with `slab_axis`) puts the net charge on the outermost layers, so that no
  field remains inside the slab.
- `centering = structure` (default with `slab_axis`) subtracts the species means of each structure.
- `slab_scale` scales the long-range slab term separately from the short-range couplings, e.g. for its screening.

## Quantum dot (`quantum_dot/`)

A Cs₂₀Pb₈Br₃₆ dot in a cubic 40 Å box, Γ point only. The steps run in folders `01` to `05`; later steps read the
parameter files of earlier ones (`../02/params.dat`, `../04/ml_params.dat`).

| File | Step |
|---|---|
| `hconf_01_tb_alpha_hyperopt` | radial parameters `alpha` of the TB with the PME on-site term (data without SOC) |
| `hconf_02_tb_pme_fit` | TB and Ewald couplings with the `alpha` values of step 1 (data without SOC) |
| `hconf_03_ml_hyperopt` | ML hyperparameters with descriptors of the distorted frames (data with SOC) |
| `hconf_04_soc_ml_fit` | SOC and ML fit with the hyperparameters of step 3 |
| `hconf_05_eval` | eigenvalues of new frames with the final model |

- `cluster = true` adds the dipole correction of the periodic images of the dot (vacuum instead of tinfoil boundary
  conditions). It needs `method = ewald` and a box close to cubic.
- The effective charges of a non-stoichiometric dot need not add up to zero (here +3.17 e: 12 CsBr units more than
  CsPbBr₃ with `q_Cs + q_Br ≠ 0`). With `cluster`, `charge_compensation = global` (the default) puts the net charge as
  a uniform background on the atoms, so that the Ewald sum of the box is neutral. Do not use `local` for clusters: it
  compares with the most common coordination of each species, which in a dot is that of the surface.
- For small dots, `method = wolf` with an `rcut` larger than the dot (and a vacuum wider than `rcut`) is also the
  potential of the isolated dot and needs no `cluster`.
- `apply_distortion = true` in the ML block: without it, the descriptors use the reference geometry for every frame,
  and the ML correction cannot follow the distortions that drive the gap fluctuations.
- The evaluation must use the same descriptor settings as the fit (`apply_distortion`, `env_scale`, `sim_params`).

## Band window (`bandmin`)

`bandmin` is the first DFT band that the TB basis describes: the number of semicore bands plus one. With the basis
Cs s, Pb s p, Br p, the semicore states are Cs 5s 5p (4 per Cs), Pb 5d (5 per Pb) and Br 4s (1 per Br):

- without SOC: `bandmin = 4 N_Cs + 5 N_Pb + N_Br + 1`, with SOC twice the semicore count plus one;
- the VBM is band `N_occupied - (bandmin - 1)` of the window.

For Cs₂₀Pb₈Br₃₆: 156 semicore bands, `bandmin = 157` (VBM = band 116 of the window), with SOC `bandmin = 313`
(VBM = band 232). The Pb 6s states lie among the semicore states in energy but belong to the TB basis, so they are
inside the window. Check the window on a relaxed structure before a fit: an index that is off by a few bands forces
valence states of the model onto conduction-band data.

## Data

- The k-points stored with the eigenvalues must be the ones of the calculation (one Γ point for the dot).
- The atom order of `POSCAR` must be that of `structures.h5` (or `XDATCAR`).
- Training, validation and test frames should not overlap; test on frames of another trajectory if possible.
