# [Molecular Dynamics](@id md-run)

Hamster can run molecular dynamics (MD) with energies and forces from its own Hamiltonian. The potential energy is the
free energy of the electrons in the Hamster Hamiltonian plus a learned nuclear term:

```math
E(\{\mathbf{R}\}) = \underbrace{\Omega(\mu) + \mu N_e}_{\text{electrons}} + E_\text{rep}(\{\mathbf{R}\}),
\qquad \Omega(\mu) = -g_s k T \, \mathrm{Tr} \ln\left(1 + e^{-(H - \mu)/kT}\right).
```

``\Omega + \mu N_e`` is the Mermin free energy of ``N_e`` electrons at the electronic temperature ``kT`` (`kt`), with
the spin degeneracy ``g_s`` (2 without SOC). Its derivative with respect to the positions is the Hellmann-Feynman force
``\mathbf{F}_I = -g_s \mathrm{Tr}[f(H)\, \partial H / \partial \mathbf{R}_I]`` with the Fermi function ``f``, which needs
only the density matrix on the pairs of atoms that are coupled by the Hamiltonian. The nuclear term ``E_\text{rep}``
contains everything the band energy misses (ion-ion repulsion, double counting, dispersion) and is learned from DFT
energies and forces.

The MD block switches on an MD calculation. Its `task` selects one of three steps:

1. `fit`: learn the nuclear term from DFT data (e.g. VASP MD runs of a small cell).
2. `eval`: energies and forces of other DFT data with the fitted model.
3. `run`: NVT molecular dynamics, e.g. of a large supercell, with an analysis of the trajectory.

A `HyperOpt` block together with `task = fit` optimizes the hyperparameters of the fit for the validation data.

```@docs
Hamster.run_calculation(::Val{:md}, ::Any, ::Config)
```

## The Hamiltonian in the MD

The MD uses the TB model (with `init_params` and `rllm_file`), the Wolf/Zahn onsite corrections of the `Ewald` block,
and SOC (`soc = true`). The ML corrections (`ML` block) are not supported yet, because their derivatives with respect
to the positions are not implemented.

- **Pairs:** as in the MD mode of standard runs, the TB pairs are the ones of the reference structure (`poscar`) within
  `rcut`, and the matrix elements are evaluated at the actual positions, with the cutoff function of `rcut_tol`. Pairs
  that are farther than `rcut` in the reference structure are never included, since their parameters are usually not
  fitted. The MD is therefore meant for solids whose atoms vibrate around their reference sites; Hamster warns if an
  atom moves farther than 1.5 Å from its site.
- **Matrix elements:** the distance dependence is the spline of `rllm.dat` times the cutoff function, the angular part
  is the Slater-Koster factor of the geometry tensor. The angular parts are sampled from the geometry tensor and
  represented exactly by polynomials of the bond direction, so the MD Hamiltonian equals the one of a standard run of the
  same structure, and its derivatives are analytic.
- **Electrostatics:** ``\Delta\varepsilon_i = -\gamma_s(\phi_i - \bar\phi_s)`` with the Wolf or Zahn potentials of the
  effective charges and the mean ``\bar\phi_s`` of species ``s`` in the structure (as for a single structure in a
  standard run). `method = ewald` is not supported in the MD (no forces of the reciprocal-space sum); use `wolf`.
- **Electrons:** the number of electrons is the sum of the `valence` tags of the atom blocks (electrons that an atom
  contributes to the TB bands, e.g. Cs 0, Pb 2, Br 6 for CsPbBr3) minus `charge`.
- `sepNN = true` is not supported, since the separate nearest-neighbor parameters switch discontinuously with the
  distance.

## Electronic solvers

- `electrons = exact`: dense diagonalization at the k-points of `kgrid` (Γ-centered, all points of the grid). The
  cost grows with the cube of the number of orbitals. Use it for the small cells of the DFT reference data, with the
  k-points of the DFT calculation.
- `electrons = kpm`: linear scaling at Γ. The grand potential and the density matrix are expanded in Chebyshev
  polynomials of the Hamiltonian (Fermi operator expansion). Traces and the elements of the density matrix on the TB
  pairs are obtained with probing vectors: the atoms of a species are colored such that atoms of the same color are
  farther apart than `probe_radius`, and every color and orbital gets one vector. The number of vectors depends on the
  probing radius, not on the size of the cell, so the cost grows linearly with the number of atoms. The density matrix
  is the exact derivative of the polynomial grand potential, so forces and energies are consistent for any expansion
  order, and the energy is conserved in NVE runs. The recursion runs on the CPU threads or, with `device = gpu`, on the
  GPU (`kpm_precision = single` is faster on GPUs with few double-precision units).
- `electrons = auto` (default): exact up to `exact_max_orbitals` orbitals, kpm above.

Practical advice:

- KPM works at Γ only, so the MD cell has to be large enough that Γ represents the Brillouin zone, e.g. at least the
  supercell that corresponds to the k-point grid of the training data (2×2×2 cell with 3×3×3 k-points → 6×6×6 cell).
- The expansion order grows as (spectral width) / `kt`; `kt` = 0.1 eV is a good start for semiconductors. Larger
  values are cheaper, but smear the occupations across the gap.
- Check the probing radius once for a structure of the run: compare energies and forces of `electrons = kpm` with
  the exact solver at Γ (e.g. in a smaller supercell) for increasing `probe_radius`. The error decreases exponentially
  with the radius in gapped systems, but drops in steps where the radius passes the distance of equivalent atoms along
  chains of strong bonds: in perovskites ABX₃, the B–X–B chains carry the slowest decay, so the radius should exceed
  twice the lattice constant.
- The fit and the run should use the same electrons (`kt`), and the same Hamiltonian settings.

## Nuclear term

```math
E = \alpha E_\text{el} + \sum_s e_s N_s + \sum_{i<j} \phi_{s_i s_j}(r_{ij})
  + \sum_i \sum_{j<k} \sum_l c_{s_i; s_j s_k, l}\, P_l(\cos\theta_{jik})\, w(r_{ij})\, w(r_{ik}) + \lambda_C E_C + E_\text{wall}
```

- ``\phi_{AB}(r) = \sum_k c_{ABk} B_k(r)``: one pair potential per species pair from `rep_nknots` cubic B-splines
  between `rep_rmin` and `rep_rcut`, which vanish smoothly at `rep_rcut`.
- Three-body terms: Legendre polynomials up to `rep_lmax3` of the angle between two neighbors j, k of atom i (closer
  than `rep_rcut3`), with ``w(r) = (1 - r/r_\text{cut3})^3``, one set of coefficients per species triplet. They describe
  bond-angle forces (e.g. octahedral tilts) that pair potentials cannot. `rep_lmax3 = -1` switches them off.
- ``E_C``: Coulomb energy of the effective charges `qeff` with the Wolf/Zahn kernel of the `Ewald` block, with a
  fitted scale ``\lambda_C`` (`rep_coulomb`).
- ``e_s``: energy per atom of every species (the reference energies of the DFT code).
- ``E_\text{wall}``: a fixed cubic repulsion below `rep_rmin` (by default the shortest distance of the pair in the
  training data minus `rep_margin`), where no data constrain the splines.
- ``\alpha``: 1, or fitted with `rep_electron_scale = true`. TB models fitted to band structures do not necessarily
  give accurate Hellmann-Feynman forces; a fitted ``\alpha`` weights them down. If the nuclear term is flexible (many
  three-body terms), the fit may weight them down almost completely, which turns the model into a classical potential
  with the Hamiltonian evaluated alongside; check the fitted value (in `hamster.out`) and compare with ``\alpha = 1``.

The term is linear in its coefficients, which are fitted by linear least squares to the differences between the DFT
and the electronic energies and forces: mean squared errors of the energies per atom and of the force components with
the weights `energy_weight` and `force_weight`, ridge regularization `lambda` and a penalty `smoothness` on the second
differences of the spline coefficients. The electronic energies and forces of the data are stored in `cache` and
reused, e.g. for different nuclear terms in a hyperparameter optimization.

Advice:

- Split the reference data by time (e.g. the last part of every DFT run as validation data), since consecutive MD
  frames are correlated.
- Pair potentials alone are usually not enough; the three-body terms with a cutoff that includes the second
  neighbors help most. Optimize `rep_rcut`, `rep_nknots`, `rep_rcut3`, `rep_lmax3` and `smoothness` with a
  `HyperOpt` block.
- Splines of pairs that never come close in the data are only constrained by the regularization; increase
  `smoothness` if they show unphysical features.
- The validation error alone does not show whether the model is stable away from the data: without regularization
  (`lambda`), the many three-body coefficients can cancel on the data and create holes elsewhere, into which MD runs
  fall after some picoseconds (the potential energy drops without bound). Check every candidate model with a short MD
  run above the target temperature (e.g. in the DFT cell with the exact solver) and keep a small `lambda`.

## Workflow

**1. Fit** the nuclear term to DFT MD runs (here VASP, `vasprun.xml`; HDF5 files with `positions`, `lattice`, `types`,
`energies` and `forces` work as well):

```bash
begin Options
    poscar = POSCAR              # reference structure of the DFT cell (same atom order as the data)
    rcut = 5.5
    load_rllm = true
    rllm_file = rllm.dat
    init_params = params.dat
end

begin Ewald
    method = wolf
    rcut = 15.0
    alpha = 0.2
    init_params = params.dat
end

begin MD
    task = fit
    electrons = exact
    kgrid = 3 3 3                # k-points of the DFT data
    kt = 0.1
    train_data = 425K/vasprun.xml 525K/vasprun.xml 625K/vasprun.xml
    val_data = 425K_val/vasprun.xml
    skip = 200                   # equilibration of the DFT runs
    stride = 10
    rep_rcut = 6.5
    rep_nknots = 14
end

begin atom Pb
    orbitals = s px py pz
    qeff = 1.248
    valence = 2
end
...
```

The fit writes the nuclear term to `model` (default `md_model.h5`) and the pair potentials as a table
(`md_model_pairs.dat`), and the errors to `hamster.out`. The predictions of all frames are in `hamster_out.h5` (for
parity plots).

**2. Hyperparameter optimization** (optional): add a `HyperOpt` block with parameters of the MD block, e.g.

```bash
begin HyperOpt
    params = MD_rep_rcut MD_rep_nknots MD_rep_rcut3 MD_rep_lmax3 MD_smoothness
    lowerbounds = 5.0 6 3.5 2 0.0
    upperbounds = 8.0 24 6.5 8 0.01
    stepsizes = 0.25 1 0.25 1 0.0005
    mode = tpe
    niter = 50
end
```

The loss is the validation force RMSE + `loss_energy_weight` × validation energy RMSE per atom. The electronic part is
computed once and taken from the cache in all iterations, unless an electronic setting (e.g. `MD_kt`) is optimized.

**3. Run** NVT dynamics, e.g. of a 6×6×6 supercell of the DFT cell on the GPU:

```bash
begin Options
    device = gpu
    ...
end

begin MD
    task = run
    supercell = 3 3 3
    electrons = kpm
    probe_radius = 10.0
    temperature = 425
    dt = 4.0
    nsteps = 10000
    thermostat = csvr
    tau = 100
    traj_every = 5
    eig_every = 100
    equilibration = 1000
end
```

## Output of a run

The output file (`output`, default `md.h5`) contains:

- `thermo/`: step, time (fs), temperature, potential, kinetic and conserved energy (with the energy exchanged with
  the thermostat), electronic and nuclear energy, chemical potential and pressure (GPa), every `thermo_every` steps;
- `trajectory/`: unwrapped Cartesian positions (Å) and velocities (Å/fs) every `traj_every` steps;
- `positions`, `lattice`, `types`: the trajectory in fractional coordinates, so the file can be used as `xdatcar` of a
  standard run (with the same `poscar`) to compute band structures, DOS or other quantities of the snapshots with any
  Hamster model, including ML corrections;
- `eigenvalues/`: eigenvalues every `eig_every` steps (exact: all eigenvalues at all k-points; kpm: `neig` eigenvalues
  at Γ by shift-invert, half of them at each edge of the gap found in the electron count);
- `analysis/`: mean and standard deviation of temperature and pressure, partial radial distribution functions,
  vibrational density of states per species (mass-weighted power spectrum of the velocities, normalized to 3 per atom),
  velocity autocorrelation and mean squared displacements, for the frames after `equilibration` steps;
- `restart/`: positions, velocities, step and thermostat state of the last write. With `restart = true`, the run
  continues from there and appends to the file; the random numbers of the thermostats depend only on `seed` and the
  step, so a restarted run gives the same trajectory as an uninterrupted one.

The analysis functions (`md_analysis`, `md_rdf`, `md_vdos`, `md_msd`) can also be called on any output file.

## Thermostats

- `csvr` (default): stochastic velocity rescaling (Bussi, Donadio, Parrinello 2007), applied for half a step before and
  after every velocity-Verlet step. It samples the canonical ensemble and changes the dynamics only slightly, so
  vibrational spectra can be taken from thermostatted runs (`tau` of a few hundred fs).
- `langevin`: BAOAB Langevin dynamics with the friction 1/`tau`.
- `none`: NVE (velocity Verlet), e.g. to check the energy conservation and the time step.

## Tags

See [MD tags](@ref md-tags).
