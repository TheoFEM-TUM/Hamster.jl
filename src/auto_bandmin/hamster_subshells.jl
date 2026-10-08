#using PeriodicTable

const L = Dict(
    's' => 0,
    'p' => 1,
    'd' => 2,
    'f' => 3,
)

const SUPERSCRIPT_DIGITS = Dict(
    '⁰' => '0',
    '¹' => '1',
    '²' => '2',
    '³' => '3',
    '⁴' => '4',
    '⁵' => '5',
    '⁶' => '6',
    '⁷' => '7',
    '⁸' => '8',
    '⁹' => '9',
)

function occupied_subshells(element)
    [
        (
            parse(Int, m.captures[1]),
            L[m.captures[2][1]],
            parse(Int, String([SUPERSCRIPT_DIGITS[c] for c in m.captures[3]])),
        )
        for m in eachmatch(r"(\d+)([spdf])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", element.el_config)
    ]
end

"""
    chemical_valence_orbitals(Z::Integer)
    chemical_valence_orbitals(symbol::AbstractString)

Return chemically relevant valence orbitals as `(n, l, m)` tuples.
For common organic elements (H, C, N, O, P, S, and halogens), fully occupied
`(n, l)` subshells are excluded.

Convention:
- s-block: outer ns
- p-block: outer ns and np
- halogens (group 17): outer np only
- d-block: outer ns and (n-1)d
- f-block: outer ns, (n-1)d, and (n-2)f when occupied
"""
function chemical_valence_orbitals(Z::Integer)
    element = elements[Z]

    nmax = findlast(!iszero, element.shells)
    group = element.xpos
    occupied = occupied_subshells(element)
    occupancies = Dict((n, l) => occupancy for (n, l, occupancy) in occupied)

    valence_subshells =
        if group <= 2
            # s-block
            [(n, l) for (n, l, _) in occupied
                if n == nmax && l == 0]

        elseif group <= 12
            # d-block
            [(n, l) for (n, l, _) in occupied
                if (n == nmax && l == 0) ||
                   (n == nmax - 1 && l == 2)]

        elseif group <= 16
            # p-block, groups 13–16
            [(n, l) for (n, l, _) in occupied
                if n == nmax && l <= 1]

        elseif group == 17
            # Halogens: np only, exclude ns²
            [(n, l) for (n, l, _) in occupied
                if n == nmax && l == 1]

        elseif group == 18
            # Noble gases: np only
            [(n, l) for (n, l, _) in occupied
                if n == nmax && l == 1]

        else
            # f-block
            [(n, l) for (n, l, _) in occupied
                if (n == nmax     && l == 0) ||
                   (n == nmax - 1 && l == 2) ||
                   (n == nmax - 2 && l == 3)]
        end

    if element.number in (1, 6, 7, 8, 15, 16) || group == 17
        filter!(
            subshell -> occupancies[subshell] < 2 * (2 * subshell[2] + 1),
            valence_subshells,
        )
    end

    # Expand each subshell into its m_l orbitals
    orbitals = NTuple{3,Int}[]

    for (n, l) in valence_subshells
        for m in -l:l
            push!(orbitals, (n, l, m))
        end
    end

    return orbitals, valence_subshells
end

function chemical_valence_orbitals(symbol::AbstractString)
    element_symbol = Symbol(uppercasefirst(lowercase(strip(symbol))))
    haskey(elements, element_symbol) ||
        throw(ArgumentError("Unknown element symbol: $symbol"))
    element = elements[element_symbol]
    return chemical_valence_orbitals(element.number)
end
