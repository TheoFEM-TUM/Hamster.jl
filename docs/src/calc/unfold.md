# [Band Unfolding](@id unfold-post-processing)

The band structure of a supercell is folded into the small Brillouin zone of the supercell. Band unfolding projects the
supercell eigenstates onto the Bloch states of a primitive cell, which gives the effective band structure (spectral
function) of distorted or disordered supercells, e.g., of MD snapshots, in the primitive Brillouin zone. Hamster unfolds
the bands as a post-processing step of a [standard run](@ref standard-run).

## Usage

Add an `Unfold` block with the primitive cell and its k-points (see [Unfold tags](@ref unfold-tags)):

```bash
begin Unfold
    poscar = POSCAR_pc       # primitive cell (Direct coordinates)
    kpoints = EIGENVAL_pc    # k-points of the primitive cell (EIGENVAL or .h5 file with kpoints)
    sigma = 0.05             # Gaussian broadening (eV) of the spectral function
end
```

The supercell must be a multiple of the primitive cell (`A_sc = A_pc M` with an integer matrix `M`). Every supercell
atom is mapped onto the site of the primitive cell whose periodic image is closest (at most `tol` Å away), using the
reference positions of the structure (for MD structures: the reference POSCAR of the `Supercell` block), so distorted
structures are unfolded with the lattice of the ideal supercell. Every site holds at most one atom (the atoms are
assigned closest first). With `displaced_mapping = true`, the displaced positions of every MD structure are mapped
instead, so that defects that move during the trajectory are found in every snapshot; increase `tol` then (thermal
displacements can exceed the default of 0.5 Å; up to about half the shortest distance between sites).

Two methods are available (`method`):

- `exact` (default): the spectral weights of the eigenstates (see below), with the diagonalization of the run.
- `kpm`: the kernel polynomial method computes the spectral function of all states directly, with sparse
  matrix-vector products only (linear scaling, no diagonalization), for supercells with 10⁵ orbitals and more. The
  energy resolution is about π × bandwidth / (2 × `kpm_moments`).

The diagonalization settings of the run are used: with `sp_diag = false`, all states are unfolded (full
diagonalization), or only those in the energy window `diag_emin`, `diag_emax` of the `Options` block (faster, since
the eigenvectors outside the window are not computed); with `sp_diag = true`, the `neig` states closest to
`eig_target`, e.g., the band edges of large supercells. The spectral weights need orthonormal eigenvectors, also for degenerate states: `diag_method = shift-invert-arpack`
(ARPACK) is therefore replaced by `shift-invert` (KrylovKit, the default; with a warning). `device = gpu` is supported as well.

## Method

The Bloch state of the primitive orbital `j` at the primitive k-point `k` is, in the supercell basis at `K = Mᵀk`,
a sum over the `Nc` supercell orbitals `i` that are copies of `j`, translated by the primitive lattice vectors `T_i`.
The spectral weight of the supercell state `ψ_N` is

```math
W_N(\mathbf{k}) = \frac{1}{N_c} \sum_j \Big| \sum_{i \in j} e^{2\pi i \, \mathbf{k} \cdot \mathbf{T}_i} c_{iN} \Big|^2,
```

which needs a single pass over every eigenvector (`O(Nε)` per state). The weights of all states at `k` add up to the
number of primitive orbitals (full diagonalization); for an ideal supercell, they are 0 or 1 and the states with weight
1 are the bands of the primitive cell. Primitive k-points that fold onto the same supercell k-point are diagonalized only
once, the supercell k-points are diagonalized in batches (as many as threads, limited by the memory of the
eigenvectors), and the structures are distributed over the MPI ranks.

The phase follows the convention of the Hamiltonian in Hamster: the orbital `j` of `H(R)` sits at `r_j - R`, and
`H(k) = Σ_R H(R) exp(2πi R·k)`, which is `H(-k)` of the common convention with the orbital at `r_j + R`. The eigenvalues
are the same (time reversal), the eigenvectors at `k` are those of `-k` in the common convention.

The spectral function of the kpm method is the Chebyshev expansion (with the Jackson kernel) of
`A(k, E) = Σ_j ⟨k, j| δ(E - H(K)) |k, j⟩`, with the primitive Bloch states as deterministic starting vectors (no
stochastic error). `kpm_moments` moments need `kpm_moments / 2` products with the Hamiltonian per k-point.

## Defects, Dopants and Surfaces

The unfolding projects onto the sites of the primitive cell, so defective supercells can be unfolded as well:

- substitutional dopants and antisites: the orbitals of the atom are assigned to orbitals of the same type of the site
  (orbitals without a counterpart are not mapped),
- vacancies: the vacant sites do not contribute,
- interstitials and adsorbates (farther than `tol` from all sites, or close to a site that already holds a closer atom,
  e.g., the second atom of a split interstitial): not mapped.

The defects are found from the positions alone: a POSCAR that simply misses an atom or contains an extra one is enough.

The spectral weights then measure the host character of the states: localized defect states have small weights, and
the weights at a k-point add up to less than the number of primitive orbitals. The numbers of substituted, unmapped atoms
and vacant sites are printed and written to the output file.

For surfaces, use the 1×1 slab (with its vacuum) as the primitive cell: supercells of the slab (e.g., reconstructions or
adsorbates on a 2×2 slab) are unfolded onto the in-plane Brillouin zone of the slab.

## Effective Band Structure

From the spectral function averaged over all structures (e.g., the snapshots of an MD trajectory), the effective
bands are extracted (`effective_bands = true`): the integrated spectral function `N_k(E) = ∫ A(k, E') dE'` counts the
primitive bands, and band `n` holds the weight between `n - 1` and `n`. Its energy is the median of this weight
(`N_k = n - 1/2`), and `lower` / `upper` are the energies where `N_k = n - 1 + 0.159` and `n - 1 + 0.841` (the central
68.2%, ±1σ for a Gaussian band): the thermal broadening of the band. This needs all states (full diagonalization or kpm,
default energy grid) and a structure without vacancies or unmapped orbitals.

In a distorted supercell, the weight below a gap is not an integer at every k: weight moves between the k-points, and
only the total over all k is conserved (for CsPbBr3 MD snapshots, `N_k` in the gap varies between 19.8 and 20.3).
Counting such a `N_k` directly would put band edges and ribbons into the gap. Therefore `N_k` is set to the nearest
integer on every gap (`A < 0.01` states/eV over at least 0.1 eV) and rescaled linearly in between.

## Band-Resolved Unfolding

The counting of the effective bands needs separated bands: where bands overlap (e.g. the valence bands of a thermal
average) or a defect level adds weight, the counted bands are quantiles of the total weight, not individual bands, and
a defect level shifts the bands above it. With `band_resolved = true`, the supercell states are projected onto the band
states of the ideal primitive cell instead,

```math
|\chi_{n\mathbf{k}}\rangle = \sum_j u_{n\mathbf{k}, j} |\mathbf{k}, j\rangle, \qquad
A_n(\mathbf{k}, E) = \sum_N |\langle \chi_{n\mathbf{k}} | \psi_N \rangle|^2 \, g(E - E_N),
```

where ``u_{n\mathbf{k}}`` are the eigenvectors of the primitive cell (`poscar` of the `Unfold` block), computed with the
model of the run. Every band then has its own spectral function, whose median, central 68.2%, mean and standard
deviation are its energy and width, also where bands overlap. The sum over all bands is the spectral function of the
counting (the band states span the same space as the orbital Bloch states). For the ideal supercell, every band is one
supercell state at the energy of the primitive band (exact, without the splitting of degenerate bands by the
counting). Degenerate primitive bands (e.g. Kramers pairs) are averaged, since their eigenvectors are not unique.

With `method = kpm`, the band states are the starting vectors (instead of the orbital Bloch states), so the cost is
the same; the weight, mean energy and standard deviation of every band are computed from its first three moments
(exact, without the kernel broadening): ``\langle H \rangle = c + h \mu_1/\mu_0`` and
``\langle H^2 \rangle = c^2 + 2ch\, \mu_1/\mu_0 + h^2 (\mu_2 + \mu_0)/(2\mu_0)``. With defects, the reference is the band structure of the ideal
crystal, so the weight of a band shows how much of it survives, and the weight that a defect level takes from the host
bands is attributed to the bands it comes from.

## Output

The file `filename` (default `unfold.h5`) contains

- `kpoints`: the primitive k-points,
- `energies`: the energy grid,
- `spectral_function`: `A(E, k) = Σ_N W_N(k) g(E - E_N(k))` (energies × k-points), averaged over all structures, with a
  normalized Gaussian `g` of width `sigma` (exact) or the Jackson kernel (kpm),
- `effective_bands/energies`, `effective_bands/lower`, `effective_bands/upper`: the effective bands (bands ×
  k-points),
- `band_resolved/` (with `band_resolved = true`): `spectral_function` (energies × bands × k-points), `primitive_bands`
  (the bands of the ideal primitive cell), and for every band and k-point `median`, `lower`, `upper` (central 68.2%),
  `mean`, `std` and `weight` (its total spectral weight: 1 without defects),
- one group per structure (with `write_weights = true`): the eigenvalues `Es` and spectral weights `Ws` (states ×
  k-points) of the exact method, or the spectral function and the Chebyshev `moments` of the kpm method; with
  `band_resolved`, also `band_weights` (bands × states × k-points) or `band_moments`.
