"""
**alpha**=0.7*Z

The `alpha::Float` value determines how rapid the orbital overlap for a specific ion type falls off with distance. Defaults to 70% of the core charge.
"""
get_alpha(conf::Config, type)::Float64 = conf("alpha", type) == "default" ? 0.70 * elements[Symbol(type)].number : conf("alpha", type)

"""
    get_alpha(conf, type, l)

Decay `α` of the radial functions of the orbitals with angular momentum `l` (0, 1, 2) of the species `type`: `alpha_s`,
`alpha_p` or `alpha_d` of the atom block, else `alpha`.
"""
function get_alpha(conf::Config, type, l)::Float64
    key = "alpha_" * "spdf"[l+1]
    return conf(key, type) == "default" ? get_alpha(conf, type) : conf(key, type)
end

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
**qeff_<element>**=qeff

Site-dependent effective charge: `qeff_<element>` in the atom block of a species sets the charge of the atoms of this
species whose nearest neighbour among the elements of its `qeff_<element>` keys is `<element>` (within `site_rcut` of the
Ewald block), e.g. `qeff_N = 0.349` and `qeff_C = 0.120` in the block of H for the H atoms bonded to N and C. Atoms
without such a neighbour keep `qeff`. Returns the element of the key `qeff_<element>`, or `nothing` for other keys.
See `get_site_charges`.
"""
function site_charge_element(key)
    m = match(r"^qeff_([a-z]{1,2})$", lowercase(key))
    isnothing(m) && return nothing
    element = uppercasefirst(m[1])
    return haskey(elements, Symbol(element)) ? element : nothing
end

"""
    get_site_qeff(conf, type) -> Vector{Pair{String, Float64}}

The site-dependent charges `qeff_<element>` of the species `type` as pairs `element => charge`, sorted by element
(empty if the atom block has none).
"""
function get_site_qeff(conf::Config, type)
    haskey(conf.blocks, type) || return Pair{String, Float64}[]
    rules = Pair{String, Float64}[]
    for key in keys(conf.blocks[type])
        element = site_charge_element(key)
        isnothing(element) || push!(rules, element => Float64(conf(key, type)))
    end
    return sort(rules)
end

"""
**charge_scale**=charge_scale of the Ewald block

The `charge_scale::Float` is the coupling of the onsite energies of the ion type to the electrostatic potential of the
Ewald model. Defaults to the `charge_scale` of the `Ewald` block.
"""
get_charge_scale(conf::Config, type)::Float64 = conf("charge_scale", type) == "default" ? get_ewald_charge_scale(conf) : conf("charge_scale", type)

"""
    get_charge_scale(conf, type, l)

Coupling of the on-site energies of the orbitals with angular momentum `l` (0, 1, 2) of the species `type` to the
electrostatic potential: `charge_scale_s`, `charge_scale_p` or `charge_scale_d` of the atom block, else `charge_scale`
of the atom block, else `charge_scale` of the `Ewald` block. `l = -1` (one coupling for all orbitals, `coupling = species`):
`charge_scale` of the atom block, else of the `Ewald` block.
"""
function get_charge_scale(conf::Config, type, l)::Float64
    l < 0 && return get_charge_scale(conf, type)
    key = "charge_scale_" * "spdf"[l+1]
    return conf(key, type) == "default" ? get_charge_scale(conf, type) : conf(key, type)
end

"""
**polarizability**=0 (not polarizable)

Polarizability (Å³) of the atoms of the species in the Ewald model (`method = wolf` or `zahn`): the field of the
effective charges induces the dipole `polarizability * E / ke` on every atom of the species, whose potential is added to
the electrostatic potential of all atoms (first order: the dipoles do not polarize each other). Species without the tag
are not polarizable; set it (also to 0) to fit the polarizability with `update_polarizability`.
"""
get_polarizability(conf::Config, type)::Float64 = conf("polarizability", type) == "default" ? 0.0 : conf("polarizability", type)

"""
**nnaxes**=false

If `nnaxes::Bool=true`, the orbital axes are rotated along the connecting vectors with the nearest neighbors of the respective orbitals. The number of nearest neighbors depends on the number of orbitals.
"""
get_nnaxes(conf::Config, type)::Bool = conf("nnaxes", type) == "default" ? false : conf("nnaxes", type)

@configtag alpha Float64 0.0 "decay of the radial functions of the species (effective charge of the hydrogen-like radial functions), which sets how fast the matrix elements fall off with distance." "atom" per_atom=true getter=:custom default_doc="0.7 × atomic number" category="basis" see_also=["n"]
@configtag alpha_s Float64 0.0 "decay of the radial functions of the s orbitals of the species (hydrogen-like effective charge); overrides alpha for s orbitals." "atom" per_atom=true getter=:custom default_doc="alpha of the atom block" category="basis" see_also=["alpha"]
@configtag alpha_p Float64 0.0 "decay of the radial functions of the p orbitals of the species (hydrogen-like effective charge); overrides alpha for p orbitals." "atom" per_atom=true getter=:custom default_doc="alpha of the atom block" category="basis" see_also=["alpha"]
@configtag alpha_d Float64 0.0 "decay of the radial functions of the d orbitals of the species (hydrogen-like effective charge); overrides alpha for d orbitals." "atom" per_atom=true getter=:custom default_doc="alpha of the atom block" category="basis" see_also=["alpha"]
@configtag n Int64 0 "principal quantum number (order of the polynomial) of the radial functions of the species." "atom" per_atom=true getter=:custom default_doc="period of the element" category="basis" see_also=["alpha"]
@configtag nnaxes Bool false "orient the orbital axes along the bonds to the nearest neighbours (the number of neighbours depends on the number of orbitals)." "atom" per_atom=true getter=:custom category="basis"
@configtag qeff Float64 0.0 "effective charge of the species for the Ewald model; qeff_<element> (e.g. qeff_N = 0.349 in the block of H) sets the charge of the atoms bonded to <element>." "atom" per_atom=true getter=:custom category="electrostatics" see_also=["charge_scale", "site_rcut"]
@configtag charge_scale Float64 0.0 "coupling of the on-site energies of the species to the electrostatic potential." "atom" per_atom=true getter=:custom default_doc="charge_scale of the Ewald block" category="electrostatics" see_also=["qeff"]
@configtag charge_scale_s Float64 0.0 "coupling of the on-site energies of the s orbitals of the species to the electrostatic potential (coupling = orbital in the Ewald block)." "atom" per_atom=true getter=:custom default_doc="charge_scale of the atom block" category="electrostatics" see_also=["charge_scale"]
@configtag charge_scale_p Float64 0.0 "coupling of the on-site energies of the p orbitals of the species to the electrostatic potential (coupling = orbital in the Ewald block)." "atom" per_atom=true getter=:custom default_doc="charge_scale of the atom block" category="electrostatics" see_also=["charge_scale"]
@configtag charge_scale_d Float64 0.0 "coupling of the on-site energies of the d orbitals of the species to the electrostatic potential (coupling = orbital in the Ewald block)." "atom" per_atom=true getter=:custom default_doc="charge_scale of the atom block" category="electrostatics" see_also=["charge_scale"]
@configtag polarizability Float64 0.0 "polarizability (Å³) of the species: first-order dipoles induced by the field of the effective charges add their potential to the electrostatic on-site model (method = wolf or zahn)." "atom" per_atom=true getter=:custom default_doc="0 (not polarizable)" category="electrostatics" see_also=["qeff", "update_polarizability"]
