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
"""
**valence**=0

The `valence::Float` is the number of electrons that an atom of this type contributes to the TB bands, used by the MD
(`MD` block) to fill the bands, e.g. Cs 0, Pb 2 (6s), Br 6 (4p) for CsPbBr3.
"""
get_valence(conf::Config, type)::Float64 = conf("valence", type) == "default" ? 0. : conf("valence", type)
