# [DOS, PDOS, COHP and JDOS](@id dos-post-processing)

Hamster can compute the density of states (DOS), the projected DOS (PDOS), crystal orbital Hamilton populations (COHP)
and the joint density of states (JDOS) as a post-processing step of a [standard run](@ref standard-run). The quantities are computed from the real-space
Hamiltonian of every structure, so no additional input is needed.

## Usage

Add the `dos` tag to the `Options` block:

```bash
begin Options
    dos = pdos cohp      # any combination of dos, pdos, cohp and jdos; dos = all computes dos, pdos and cohp
end
```

All other settings are optional and go into a `DOS` block (see [DOS tags](@ref dos-tags)), for example:

```bash
begin DOS
    projection = element    # element (Pb, Br) or orbital (Pb-s, Pb-px, ...)
    cohp_pairs = Pb-Br      # default: all coupled pairs
    nelec = 8               # electrons per structure (here: GaAs primitive cell): Fermi energy, IDOS and ICOHP
    sigma = 0.05            # broadening (eV) of the exact method
    jdos_velocity = xyz     # with dos = jdos: also the velocity-weighted JDOS of these components
end
```

The k-points are taken from the `kpoints` tag of the run (equal weights); for large supercells, Γ (`kpoints = gamma`,
the default) is usually sufficient.

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

The JDOS is only available with the exact method (it needs the individual eigenvalues). Its grid runs from 0 to
`jdos_emax` (default: the largest transition energy + 5 `sigma`) with `ne` points and uses the same broadening as the DOS.
Since there are (occupied × empty) transitions per k-point, every transition energy is first split linearly between the
two neighbouring grid points and the result is broadened afterwards; this is accurate as long as the grid spacing is
small compared with `sigma`.

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
- **JDOS:** the density of vertical transitions between occupied and empty states,
  ```math
  \mathrm{JDOS}(\omega) = g \sum_k w_k \sum_{v, c} f_{vk} (1 - f_{ck})\, \delta\left(\omega - (E_{ck} - E_{vk})\right),
  ```
  with the zero-temperature occupations ``f`` for `nelec` electrons (required). It contains no transition matrix
  elements, so it is the spectrum of possible excitation energies, not an absorption spectrum. Spin-conserving
  transitions are counted (without SOC: twice, once per spin). For an insulator,
  ``\int \mathrm{JDOS}\, d\omega = g \sum_k w_k N_v N_c`` (occupied × empty bands).
- **Velocity-weighted JDOS:** with `jdos_velocity = xyz` (any combination of components), the JDOS is also computed
  with every transition weighted by the squared velocity matrix element,
  ```math
  J_a(\omega) = g \sum_k w_k \sum_{v, c} f_{vk} (1 - f_{ck})\, |\langle vk | \hat v_a | ck \rangle|^2\, \delta\left(\omega - (E_{ck} - E_{vk})\right)
  ```
  in (Å/fs)²/eV. The velocity operator is the one of the current operator (`write_current`),
  ``v_a(R) = -\tfrac{i}{\hbar} d_a(R) \circ H(R)`` with the bond vectors ``d``: it only contains the hopping between
  different sites, not intra-atomic dipole matrix elements (e.g. between the s and p orbitals of one atom). In the
  independent-particle approximation, the absorptive part of the dielectric function is
  ``\varepsilon_{2,aa}(\omega) \propto J_a(\omega) / \omega^2`` (divided by the cell volume).

## Output

The results are written to `dos.h5` (tag `filename` of the `DOS` block), with one group per structure (named by the
configuration index, prefixed with the system name for multiple systems):

| Dataset | Content |
|---|---|
| `energies` | energy grid (eV) |
| `dos` | total DOS (states/eV) |
| `pdos/<label>` | PDOS of every label, e.g. `pdos/Pb` |
| `cohp/<A-B>` | COHP of every pair, e.g. `cohp/Pb-Br` |
| `fermi_energy` | Fermi energy (only with `nelec`) |
| `idos`, `icohp/<A-B>` | integrated DOS and COHP on the energy grid (only with `nelec`) |
| `icohp_fermi/<A-B>` | ICOHP at the Fermi energy (only with `nelec`) |
| `jdos_energies`, `jdos` | transition-energy grid (eV) and JDOS (transitions/eV), only with `jdos` |
| `jdos_velocity/<a>` | velocity-weighted JDOS of component `a` ((Å/fs)²/eV), only with `jdos_velocity` |

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
