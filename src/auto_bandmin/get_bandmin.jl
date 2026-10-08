
"""
    remove_chemical_valence_orbitals(dft_orbitals)

For each element, remove the subshells selected by
`chemical_valence_orbitals(element)` from its DFT orbital list. Remaining
entries retain their `(n, l, occupancy)` values.
"""
function remove_chemical_valence_orbitals(
    dft_orbitals::AbstractDict{<:AbstractString, <:AbstractVector},
)
    remaining_orbitals = Dict{String, Vector{Tuple{Int, Int, Float64}}}()

    for (element, orbitals) in dft_orbitals
        _, valence_subshells = chemical_valence_orbitals(element)
        valence_set = Set(valence_subshells)
        remaining_orbitals[String(element)] = [
            (Int(n), Int(l), Float64(occupancy))
            for (n, l, occupancy) in orbitals
            if (n, l) ∉ valence_set
        ]
    end

    return remaining_orbitals
end

"""
    total_nonchemical_electrons(nonchemical_orbitals, atom_counts)

Calculate the total nonchemical-orbital electron count across the structure.
`atom_counts` maps element symbols to their number of atoms.
"""
function total_nonchemical_electrons(
    nonchemical_orbitals::AbstractDict{<:AbstractString, <:AbstractVector},
    atom_counts::AbstractDict{<:AbstractString, <:Integer},
)
    total = 0.0

    for (element, orbitals) in nonchemical_orbitals
        haskey(atom_counts, element) ||
            throw(ArgumentError("No atom count provided for $element"))
        count = atom_counts[element]
        count >= 0 ||
            throw(ArgumentError("Atom count for $element must be non-negative"))

        total += count * sum(orbital[3] for orbital in orbitals; init=0.0)
    end

    return total
end

"""
    total_nonchemical_electrons(poscar_path, potcar_path)

Read the POSCAR atom counts and POTCAR configurations, then calculate the
structure's total number of electrons in nonchemical orbitals.
"""
function total_nonchemical_electrons(
    poscar_path::AbstractString,
    potcar_path::AbstractString,
)
    poscar_lines = readlines(poscar_path)
    length(poscar_lines) >= 7 ||
        throw(ArgumentError("POSCAR must contain element and atom-count lines"))

    elements = split(strip(poscar_lines[6]))
    count_tokens = split(strip(poscar_lines[7]))
    counts = tryparse.(Int, count_tokens)
    any(isnothing, counts) &&
        throw(ArgumentError("POSCAR line 7 must contain atom counts"))
    length(elements) == length(counts) ||
        throw(ArgumentError("POSCAR element and atom-count lists have different lengths"))

    atom_counts = Dict{String, Int}(elements[i] => counts[i] for i in eachindex(elements))
    configurations, zvals = read_POTCAR(poscar_path, potcar_path)
    nonchemical_orbitals = remove_chemical_valence_orbitals(dft_subshells(configurations, zvals))

    return total_nonchemical_electrons(nonchemical_orbitals, atom_counts)
end

function get_band_min(poscar_path::AbstractString, potcar_path::AbstractString, conf=get_empty_config(); auto_bandmin = get_auto_bandmin(conf))
    if !auto_bandmin
        return get_bandmin(conf)
    else
        N_nc_elec = total_nonchemical_electrons(poscar_path, potcar_path)
        bandmin = floor(Int, 0.5 * N_nc_elec) + 1
        return bandmin
    end
end