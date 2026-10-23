# Spin-orbit coupling

A spin-orbit coupling (SOC) term is added to the Hamiltonian if a `SOC` block is present in the config file (an empty
block is enough, or `soc = true` in the `Options` block). Every orbital then becomes two spin orbitals (↑, ↓), and the
number of bands doubles.

## Model

SOC is described by an on-site term per atom,

```math
H_\mathrm{SOC} = \sum_i \gamma_{s(i)} \, \mathbf{L}_i \cdot \mathbf{S}_i ,
```

with one parameter ``\gamma_s`` (in eV) per species ``s``, written as `γ_<element>` to the parameter file. For p orbitals,
``\mathbf{L}\cdot\mathbf{S}`` has the eigenvalues ``+\tfrac{1}{2}`` (``j = 3/2``) and ``-1`` (``j = 1/2``), so the p levels of
an isolated atom split by ``\tfrac{3}{2}\gamma``. s orbitals do not contribute. The matrices are set up in the
``|l\, m\, s\rangle`` basis and transformed to the real orbitals of the basis (also to `sp3` and `sp3dr2` hybrids, using the
hybrid axes of every atom).

## Fitting

SOC parameters are fitted like TB parameters, to a DFT band structure computed with SOC (see the
[SOC tutorial](@ref tutorial-soc)):

```bash
begin Optimizer
    update_tb = false        # TB parameters from a fit without SOC stay fixed
    train_data = EIGENVAL_soc
    bandmin = 27             # with SOC, every DFT band appears twice
end

begin SOC
    update = true            # default in a fit
    init_params = ones       # start values: ones (1 eV), zeros, rand, or a parameter file
end
```

Fit the TB parameters without SOC first and add SOC afterwards; the SOC parameters can also be fitted together with the
ML corrections on MD data. In a standard run, the SOC parameters are read from `init_params` of the `SOC` block, or,
if it is not set, from the parameter file of the `Options` block.

## Settings

See the [SOC block](@ref block-soc) for all tags.
