"""
**alpha**=0.7*Z

The `alpha::Float` value determines how rapid the orbital overlap for a specific ion type falls off with distance. Defaults to 70% of the core charge.
"""
get_alpha(conf::Config, type)::Float64 = conf("alpha", type) == "default" ? 0.70 * elements[Symbol(type)].number : conf("alpha", type)

"""
**n**=n_period

The `n::Int` value determines the order of the polynomial that is used to model the distance dependence the orbital overlap for a specific ion type. Defaults to the period the atom species belongs to.
"""
get_n(conf::Config, type)::Int64 = conf("n", type) == "default" ? elements[Symbol(type)].period : conf("n", type)

"""
**qeff**=0

The `qeff::Float` determines the effective charge of the ion for Ewald summation.
"""
get_qeff(conf::Config, type)::Float64 = conf("qeff", type) == "default" ? 0. : conf("qeff", type)

"""
**charge_scale**=charge_scale of the Ewald block

The `charge_scale::Float` is the coupling of the onsite energies of the ion type to the electrostatic potential of the
Ewald model. Defaults to the `charge_scale` of the `Ewald` block.
"""
get_charge_scale(conf::Config, type)::Float64 = conf("charge_scale", type) == "default" ? get_ewald_charge_scale(conf) : conf("charge_scale", type)

"""
**nnaxes**=false

If `nnaxes::Bool=true`, the orbital axes are rotated along the connecting vectors with the nearest neighbors of the respective orbitals. The number of nearest neighbors depends on the number of orbitals.
"""
get_nnaxes(conf::Config, type)::Bool = conf("nnaxes", type) == "default" ? false : conf("nnaxes", type)

@configtag alpha Float64 0.0 "decay of the radial functions of the species (effective charge of the hydrogen-like radial functions), which sets how fast the matrix elements fall off with distance." "atom" per_atom=true getter=:custom default_doc="0.7 × atomic number" category="basis" see_also=["n"]
@configtag n Int64 0 "principal quantum number (order of the polynomial) of the radial functions of the species." "atom" per_atom=true getter=:custom default_doc="period of the element" category="basis" see_also=["alpha"]
@configtag nnaxes Bool false "orient the orbital axes along the bonds to the nearest neighbours (the number of neighbours depends on the number of orbitals)." "atom" per_atom=true getter=:custom category="basis"
@configtag qeff Float64 0.0 "effective charge of the species for the Ewald model." "atom" per_atom=true getter=:custom category="electrostatics" see_also=["charge_scale"]
@configtag charge_scale Float64 0.0 "coupling of the on-site energies of the species to the electrostatic potential." "atom" per_atom=true getter=:custom default_doc="charge_scale of the Ewald block" category="electrostatics" see_also=["qeff"]
