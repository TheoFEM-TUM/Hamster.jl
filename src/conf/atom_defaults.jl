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

The `qeff::Float` determines the effective charge of the ion for Ewald summation. If it is not set for an ion type and
`auto_qeff = true` is set in the `Ewald` block, the group default from `get_auto_qeff` is used instead.
"""
function get_qeff(conf::Config, type)::Float64
    if conf("qeff", type) ≠ "default"
        return conf("qeff", type)
    elseif get_ewald_auto_qeff(conf)
        return get_auto_qeff(type)
    else
        return 0.
    end
end

"""
    get_auto_qeff(type) -> Float64

Default effective charge of an ion type for Ewald summation, chosen by its group in the periodic table
(one common formal oxidation state per group):

| Group | 1  | 2  | 3  | 4  | 5  | 6  | 7–10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|-------|----|----|----|----|----|----|------|----|----|----|----|----|----|----|----|
| qeff  | +1 | +2 | +3 | +4 | +5 | +4 | +2   | +1 | +2 | +3 | +2 | −3 | −2 | −1 | 0  |

Lanthanides and actinides are assigned +3.
"""
function get_auto_qeff(type)::Float64
    element = elements[Symbol(type)]
    # Lanthanides and actinides are placed in rows 9 and 10 of the periodic table layout
    element.ypos ≥ 9 && return 3.
    group_qeff = (1., 2., 3., 4., 5., 4., 2., 2., 2., 2., 1., 2., 3., 2., -3., -2., -1., 0.)
    return group_qeff[element.xpos]
end

"""
**nnaxes**=false

If `nnaxes::Bool=true`, the orbital axes are rotated along the connecting vectors with the nearest neighbors of the respective orbitals. The number of nearest neighbors depends on the number of orbitals.
"""
get_nnaxes(conf::Config, type)::Bool = conf("nnaxes", type) == "default" ? false : conf("nnaxes", type)