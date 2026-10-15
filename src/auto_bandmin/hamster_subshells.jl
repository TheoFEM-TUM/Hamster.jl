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

const DEFAULT_HAMSTER_ORBITALS = Dict{String,Tuple{Vararg{String}}}(
    "K" => ("s",),
    "Ge" => ("s", "px", "py", "pz"),
    "I" => ("px", "py", "pz"),
    "Cs" => ("s",),
    "Sr" => ("s",),
    "Ba" => ("s",),
    "Cl" => ("px", "py", "pz"),
    "Sn" => ("s", "px", "py", "pz"),
    "C" => ("s", "px", "py", "pz"),
    "N" => ("s", "px", "py", "pz"),
    "H" => ("s",),
    "Br" => ("px", "py", "pz"),
    "Ca" => ("s",),
    "Pb" => ("s", "px", "py", "pz"),
    "Rb" => ("s",),
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
For common organic elements (H, C, N, O, P, S), exception

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

"""    if element.number in (1, 6, 7, 8, 15, 16)
        filter!(
            subshell -> occupancies[subshell] < 2 * (2 * subshell[2] + 1),
            valence_subshells,
        )
    end"""

    # Expand each subshell into its m_l orbitals
    orbitals = NTuple{3,Int}[]

    for (n, l) in valence_subshells
        for m in -l:l
            push!(orbitals, (n, l, m))
        end
    end

    if element.number in (1, 6, 7, 8, 15, 16)
        orbitals = orbitals = NTuple{3,Int}[]
        valence_subshells = []
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

"""
    default_hamster_orbitals(Z::Integer)
    default_hamster_orbitals(symbol::AbstractString)

Return `(n, l, m)` orbitals and `(n, l)` subshells selected for an element in
`DEFAULT_HAMSTER_ORBITALS`. For each angular-momentum type listed in the
constant, the outermost occupied subshell of that type is expanded into all
its `m` orbitals.
"""
function default_hamster_orbitals(symbol::AbstractString)
    element_symbol = uppercasefirst(lowercase(strip(symbol)))
    element_key = Symbol(element_symbol)
    haskey(elements, element_key) ||
        throw(ArgumentError("Unknown element symbol: $symbol"))
    orbital_labels = get(DEFAULT_HAMSTER_ORBITALS, element_symbol, nothing)
    orbital_labels === nothing &&
        throw(ArgumentError("$element_symbol is not specified in DEFAULT_HAMSTER_ORBITALS"))

    angular_momenta = Int[]
    for label in orbital_labels
        orbital_type = first(label)
        haskey(L, orbital_type) ||
            throw(ArgumentError("Unknown orbital label '$label' for $element_symbol"))
        l = L[orbital_type]
        l in angular_momenta || push!(angular_momenta, l)
    end

    occupied = occupied_subshells(elements[element_key])
    subshells = Tuple{Int,Int}[]
    for l in angular_momenta
        matching_shells =
            [(n, shell_l) for (n, shell_l, _) in occupied if shell_l == l]
        isempty(matching_shells) &&
            throw(ArgumentError("$element_symbol has no occupied subshell with l=$l"))
        push!(subshells, (maximum(first, matching_shells), l))
    end

    orbitals = NTuple{3,Int}[]
    for (n, l) in subshells
        for m in -l:l
            push!(orbitals, (n, l, m))
        end
    end

    return orbitals, subshells
end

function default_hamster_orbitals(Z::Integer)
    return default_hamster_orbitals(String(elements[Z].symbol))
end
