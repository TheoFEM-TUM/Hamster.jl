# Tight-Binding Model

The tight-binding (TB) model represents the real-space Hamiltonian as a linear
combination of geometry-dependent matrix elements and scalar model parameters.

The model assumes that each Hamiltonian block associated with a lattice
translation vector ``\mathbf{R}`` can be written as

```math
H^{\mathbf{R}} = \sum_v V_v^0 \, h_v^{\mathbf{R}},
```

where ``V_v^0`` are the adjustable TB parameters and ``h_v^{\mathbf{R}}``
contains the geometry-dependent contribution for interaction channel ``v``.

In reciprocal space, the Hamiltonian is obtained by Fourier transformation,

```math
H(\mathbf{k}) =
\sum_{\mathbf{R}}
e^{i\mathbf{k}\cdot\mathbf{R}}
H^{\mathbf{R}}.
```

## Localized Basis

The TB model is formulated in a localized orbital basis. Each
orbital centered on atom $i$ can be written as a linear combination of atomic
orbitals expressed in real spherical harmonics,

```math
\psi_i(\mathbf{r}-\mathbf{R}) =
\frac{1}{A_i}
\sum_{lm}
f_{lm}(\boldsymbol{\omega}_i)
\phi_{lm}(\mathbf{r}-\boldsymbol{\tau}_i-\mathbf{R}),
```

where ``\boldsymbol{\omega}_i`` defines the orbital orientation, $A_i$ is a
normalization constant, and ``f_{lm}`` are expansion coefficients.

## Angular Dependence

The angular dependence of hopping matrix elements is handled using the
Slater-Koster idea. For a pair of atoms ``i`` and ``j``, a rotated coordinate frame
is constructed such that the local ``z'`` axis is aligned with the connecting
vector between the two atoms.

If ``U`` denotes the rotation into this local frame, the orbital orientation
transforms as

```math
\boldsymbol{\omega}_i^r =
U \boldsymbol{\omega}_i.
```

In this rotated frame, matrix elements between orbitals can be written as

```math
\langle \chi_i | \hat{H} | \chi_j \rangle
=
\frac{1}{A_i A_j}
\sum_{ll'm}
f^*_{lm}(\boldsymbol{\omega}_i^r)
f_{l'm}(\boldsymbol{\omega}_j^r)
\langle \phi_{lm} | \hat{H} | \phi_{l'm} \rangle.
```

Due to symmetry, only matrix elements with matching magnetic quantum number
$m$ contribute in the rotated frame.

The orientation-dependent prefactor is collected into a tensor

```math
C_{vij}^{\mathbf{R}}
=
f^*_{lm}(\boldsymbol{\omega}_i^r)
f_{l'm}(\boldsymbol{\omega}_j^r),
```

where ``v`` is a composite index labeling the interaction channel, for example
``ss\sigma``, ``sp\sigma``, ``pp\sigma``, or ``pp\pi``.

## Distance Dependence

Within the two-center approximation, the remaining atomic-orbital matrix
element is written as a product of an adjustable TB parameter and a
distance-dependent factor,

```math
\langle \phi_{lm} | \hat{H} | \phi_{l'm} \rangle
\approx
V_{ll'm}^0 \, V_{ll'm}(\Delta r),
```

where ``\Delta r`` is the distance between the two atomic centers.

Instead of introducing a separate flexible parametrization for each distance
dependence, the model computes

```math
V_{ll'm}(\Delta r)
=
\langle
\phi_{lm}(\mathbf{r})
|
\phi_{l'm}(\mathbf{r}-\Delta r\,\hat{\mathbf{e}}_z)
\rangle.
```

The radial part is based on hydrogen-like orbitals with an effective core
charge ``Z_{\mathrm{eff}}``. This reduces the complexity of the distance
dependence to a small number of physically interpretable hyperparameters.

The required overlap integrals are precomputed and interpolated using cubic
splines, making Hamiltonian construction efficient during training and
evaluation.

## Final Matrix Element

Combining angular and distance dependence gives the real-space TB matrix
element

```math
H_{ij}^{\mathbf{k}}
=
\frac{1}{A_i A_j}
\sum_{\mathbf{R}}
e^{i\mathbf{k}\cdot\mathbf{R}}
\sum_v
C_{vij}^{\mathbf{R}}
V_v(\Delta r)
V_v^0.
```

Equivalently, the real-space Hamiltonian block can be written as

```math
H_{ij}^{\mathbf{R}}
=
\sum_v
C_{vij}^{\mathbf{R}}
V_v(\Delta r)
V_v^0.
```

The implementation stores the geometry-dependent part

```math
h_{vij}^{\mathbf{R}}
=
C_{vij}^{\mathbf{R}}
V_v(\Delta r),
```

so that Hamiltonian construction reduces to the linear expression

```math
H^{\mathbf{R}}
=
\sum_v
V_v^0 h_v^{\mathbf{R}}.
```

## Parameter Sharing

For multiple structures, each structure may contain a local parameter ordering.
The model therefore stores a global parameter vector and a mapping from each
structure to the relevant global parameter indices.

For structure ``n``,

```math
V^{(n)} =
V[\mathrm{params\_ per\_ strc}[n]].
```

This allows different structures and bases to share the same optimized
parameters consistently.

## Spin Basis

If spin-orbit coupling or an explicit spin basis is requested, the Hamiltonian
blocks are expanded into spin space. In that case, each orbital degree of
freedom is duplicated for spin-up and spin-down components before additional
spin-dependent terms are added.

## Optimization

The TB parameters are optimized by differentiating the loss with respect to
the Hamiltonian.

Given

```math
\frac{\partial L}{\partial H_R},
```

the gradient with respect to a TB parameter ``V_v`` is

```math
\frac{\partial L}{\partial V_v}
=
\sum_R
\sum_{ij}
h_{vR,ij}
\frac{\partial L}{\partial H_{R,ij}}.
```

This is a direct consequence of the linear relation

```math
H_R =
\sum_v V_v h_{vR}.
```

Regularization terms can be added to the parameter gradient,

```math
\frac{\partial L_{\mathrm{tot}}}{\partial V_v}
=

\frac{\partial L}{\partial V_v}
+
\frac{\partial L_{\mathrm{reg}}}{\partial V_v}.
```

Only parameters marked as updateable are modified during optimization.

## Parameter Initialization

TB parameters can be initialized from:

* ones
* random values
* zeros
* an external parameter file

When parameters are read from a file, they are matched by their parameter
labels before being inserted into the model.