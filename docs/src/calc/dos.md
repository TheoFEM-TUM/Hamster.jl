# [DOS, PDOS and COHP](@id dos-post-processing)

Hamster can compute the density of states (DOS), the projected DOS (PDOS) and crystal orbital Hamilton populations (COHP)
as a post-processing step of a [standard run](@ref standard-run). The quantities are computed from the real-space
Hamiltonian of every structure, so no additional input is needed.

## Usage

Add the `dos` tag to the `Options` block:

```bash
begin Options
    dos = pdos cohp      # any combination of dos, pdos and cohp; dos = all computes all three
end
```

All other settings are optional and go into a `DOS` block (see [DOS tags](@ref dos-tags)), for example:

```bash
begin DOS
    projection = element    # element (Pb, Br), orbital (Pb-s, Pb-px, ...) or defect (see below)
    cohp_pairs = Pb-Br      # default: all coupled pairs
    nelec = 8               # electrons per structure (here: GaAs primitive cell): Fermi energy, IDOS and ICOHP
    sigma = 0.05            # broadening (eV) of the exact method
end
```

The k-points are taken from the `kpoints` tag of the run (equal weights); for large supercells, Γ (`kpoints = gamma`,
the default) is usually sufficient.

## Defect Projection

With `projection = defect`, the PDOS is projected onto the regions around the defects of the structure: the defect
atoms, their neighbour shells and the remaining (bulk) atoms, resolved by element, e.g. `defect-Br`, `shell1-Pb`,
`shell2-Cs`, `bulk-Br`. The defects are found automatically by mapping the structure onto the primitive cell
(`primitive_poscar`, or the `poscar` of an `Unfold` block), as for the [band unfolding](@ref unfold-post-processing):

- substituted atoms (dopants, antisites) and unmapped atoms (interstitials, adsorbates) are the defect atoms;
- separate defects (defect atoms and vacancies that are not neighbours) get separate regions, e.g. `defect1-Br`,
  `defect2-shell1-Pb` (`defect_separate = false`: one region for all); an atom next to several defects belongs to the
  closest;
- vacancies have no orbitals; their neighbours form the first shell;
- the shells are built outwards (`defect_shells`, default 2): an atom is a neighbour of an atom (or vacant site) of the
  previous shell if their distance is below 1.2 times the larger of their nearest-neighbour distances in the ideal crystal.

So a POSCAR that simply misses an atom or contains an extra one is enough. The displaced positions of every structure
are mapped (with `defect_tol`, default 1 Å), so a defect that moves during an MD trajectory is followed. The regions of
every atom (0: defect, n: shell n, -1: bulk) and the number of atoms of every label are written to the output, so the
PDOS per atom of the defect region can be compared with the bulk. Localized defect states appear as peaks of the
defect and shell PDOS (e.g. in the gap), and the integrated peak shows how much of the state sits on these atoms.

With `method = kpm`, the PDOS of the defect and shell labels is computed exactly with unit vectors on their orbitals
(in batches of 32, so the cost grows with the size of the defect region) instead of the stochastic trace, and the total
DOS is the sum of all labels.

## Localized States

With `projection = localized`, the regions are found from the eigenvectors instead of the geometry, so localized
states without a structural defect (e.g. polarons or self-trapped carriers in a distorted MD structure) are found as
well. The participation of the atoms is computed energy-resolved: the atom weights of all states near E (within the
broadening `sigma`) are averaged,

```math
w_a(E) = \frac{\sum_N g(E - E_N) \sum_{i \in a} |c_{iN}|^2}{\sum_N g(E - E_N)}, \qquad
\mathrm{PN}(E) = \frac{1}{\sum_a w_a(E)^2},
```

which does not depend on the choice of the eigenvectors within (nearly) degenerate states (in a supercell, folded
states of flat bands can otherwise mix into arbitrarily localized combinations). The participation number PN is the
number of atoms the states at E live on. It is divided by the number for states with the same element weights spread
uniformly over the atoms of every element, so that the participation ratio is about 1 for Bloch-like states, also if
they live on one sublattice only (e.g. Pb-p states on the Pb atoms), and small for localized states. Energy intervals
with a ratio below `localized_threshold` (default 0.1) contain localized states; the `round(PN)` atoms with the largest
weights at the maximum of the DOS of an interval form the region of that state, with neighbour shells as for defects
(`defect_shells`; one region per localized state). The window for localized states is the gap ± 1 eV with `nelec`
(all states otherwise, which also finds e.g. single-atom states split off from narrow semicore bands), or
`localized_emin`, `localized_emax`.

The participation (`participation/energies`, `pn`, `ratio`) and the energy, ratio, PN and atoms of every localized
state (`participation/peak_energies`, `peak_ratios`, `peak_pn`, `peak_atoms/<n>`) are written to the output. This needs
the eigenvectors (exact method). In small supercells, even a localized state covers a large fraction of the atoms, so
the ratio does not drop much (e.g. 0.25 for defects in a 16-atom cell); use supercells with a few hundred atoms.

## Methods

- `method = exact` diagonalizes the Hamiltonian at every k-point and broadens every eigenvalue with a Gaussian or
  Lorentzian of width `sigma`. The Fermi energy and the ICOHP are obtained from the eigenvalues and the occupied states
  directly.
- `method = kpm` uses the kernel polynomial method: the DOS, PDOS and COHP are expanded in `kpm_moments` Chebyshev
  polynomials (with the Jackson kernel), whose coefficients are estimated with `kpm_vectors` random-phase vectors. Only
  sparse matrix-vector products are needed, so this works for supercells with 10⁵ orbitals and more. The energy
  resolution is about π × bandwidth / (2 × `kpm_moments`), and the relative stochastic error about
  1/√(orbitals × `kpm_vectors`).
- `method = auto` (default) uses the exact method up to `exact_max_orbitals` orbitals (default 4000) and KPM above.

## Conventions

- **Spin:** without SOC, all quantities contain the spin degeneracy 2. With SOC (`soc = true`), every orbital is a spin
  orbital (degeneracy 1), and the PDOS of the two spin components of an orbital is summed. In both cases,
  ``\int \mathrm{DOS}(E)\, dE`` equals the number of electrons the bands can hold.
- **PDOS:** ``\mathrm{PDOS}_A(E) = g \sum_{k} w_k \sum_n \delta(E - E_{nk}) \sum_{i \in A} |c_{i n k}|^2``, where ``A`` is an
  element or an orbital type, depending on `projection`. The sum over all labels gives the DOS.
- **COHP:** for a pair of labels ``(A, B)``,
  ```math
  \mathrm{COHP}_{AB}(E) = g \sum_k w_k \sum_n \delta(E - E_{nk}) \sum_{i \in A, j \in B \text{ or } i \in B, j \in A} \mathrm{Re}\left(c^*_{ink} H_{ij}(k)\, c_{jnk}\right),
  ```
  summed over orbitals on *different* atoms; on-site terms are excluded. Pairs of the same label (e.g. Br-Br) contain
  the bonds between different atoms of that element. **Negative values are bonding**, positive values antibonding
  (plot -COHP to get the common "bonding to the right" picture). The COHP of all pairs plus the on-site contributions
  equals ``E \cdot \mathrm{DOS}(E)``.
- **ICOHP:** with `nelec`, the integrated COHP up to the Fermi energy is written for every pair, together with the
  integrated curves.

## Output

The results are written to `dos.h5` (tag `filename` of the `DOS` block), with one group per structure (named by the
configuration index, prefixed with the system name for multiple systems):

| Dataset | Content |
|---|---|
| `energies` | energy grid (eV) |
| `dos` | total DOS (states/eV) |
| `pdos/<label>` | PDOS of every label, e.g. `pdos/Pb` |
| `label_atoms/<label>` | number of atoms of every label |
| `regions` | region of every atom (`projection = defect` or `localized`: 0 defect, n shell n, -1 bulk) |
| `region_defect` | defect of every atom (1, 2, … for separate defects, 0 for the bulk) |
| `participation/...` | participation of the atoms (only `projection = localized`, see above) |
| `cohp/<A-B>` | COHP of every pair, e.g. `cohp/Pb-Br` |
| `fermi_energy` | Fermi energy (only with `nelec`) |
| `idos`, `icohp/<A-B>` | integrated DOS and COHP on the energy grid (only with `nelec`) |
| `icohp_fermi/<A-B>` | ICOHP at the Fermi energy (only with `nelec`) |

The group attributes contain the method and its settings. Reading the data, e.g. in Julia:

```julia
using HDF5
h5open("dos.h5") do f
    g = f["1"]
    E = read(g["energies"]); cohp = read(g["cohp/Pb-Br"]); E_F = read(g["fermi_energy"])
end
```

## Notes

- KPM determines the Fermi energy from the integrated DOS. For insulators, it is placed in the middle of the widest
  region without states near the electron count. For small systems with discrete spectra (e.g. a small supercell at Γ),
  several such regions may exist within the stochastic error; use the exact method there (the default for up to 4000
  orbitals).
- With MPI, the structures are distributed over the ranks as in the standard run.

## Functions

The post-processing can also be used directly, e.g. for a real-space Hamiltonian from an earlier run:

```@docs
Hamster.compute_dos
Hamster.orbital_labels
Hamster.DOSSettings
```
