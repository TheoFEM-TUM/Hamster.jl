
"""
    read_POTCAR(poscar_path, potcar_path)

Read atomic-configuration tables and `ZVAL` values for the elements listed in
a VASP5 POSCAR. Returns `(configurations, zvals)`, both keyed by element
symbol. Each configuration matrix has columns `n`, `l`, `j`, `E`, and `occ.`.
"""
function read_POTCAR(poscar_path::AbstractString, potcar_path::AbstractString)
    poscar_lines = readlines(poscar_path)
    length(poscar_lines) >= 7 ||
        throw(ArgumentError("POSCAR must contain element and atom-count lines"))

    elements = split(strip(poscar_lines[6]))
    counts = split(strip(poscar_lines[7]))
    all(token -> tryparse(Int, token) !== nothing, counts) ||
        throw(ArgumentError("POSCAR line 7 must contain atom counts"))
    isempty(elements) && throw(ArgumentError("POSCAR element list is empty"))
    length(elements) == length(counts) ||
        throw(ArgumentError("POSCAR element and atom-count lists have different lengths"))

    configurations = Dict{String, Matrix{Float64}}()
    zvals = Dict{String, Float64}()
    current_element = nothing
    potcar_lines = readlines(potcar_path)

    for i in eachindex(potcar_lines)
        vrhfin_match = match(r"VRHFIN\s*=\s*([A-Z][a-z]?)", potcar_lines[i])
        if vrhfin_match !== nothing
            current_element = vrhfin_match.captures[1]
            continue
        end

        zval_match = match(r"\bZVAL\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)", potcar_lines[i])
        if zval_match !== nothing
            current_element === nothing &&
                throw(ArgumentError("Found ZVAL before a VRHFIN element in POTCAR"))
            haskey(zvals, current_element) &&
                throw(ArgumentError("Multiple ZVAL values found for $current_element in POTCAR"))
            zvals[current_element] = parse(Float64, zval_match.captures[1])
        end

        if occursin("Atomic configuration", potcar_lines[i])
            current_element === nothing &&
                throw(ArgumentError("Found atomic configuration before a VRHFIN element in POTCAR"))
            i + 2 <= length(potcar_lines) ||
                throw(ArgumentError("Incomplete atomic configuration for $current_element"))

            count_match = match(r"^\s*(\d+)\s+entries\b", potcar_lines[i + 1])
            count_match === nothing &&
                throw(ArgumentError("Missing atomic-configuration entry count for $current_element"))
            nentries = parse(Int, count_match.captures[1])
            first_row = i + 3
            last_row = first_row + nentries - 1
            last_row <= length(potcar_lines) ||
                throw(ArgumentError("Incomplete atomic configuration for $current_element"))

            table = Matrix{Float64}(undef, nentries, 5)
            for (row, line_index) in enumerate(first_row:last_row)
                fields = split(strip(potcar_lines[line_index]))
                length(fields) == 5 ||
                    throw(ArgumentError("Malformed atomic-configuration row for $current_element at POTCAR line $line_index"))
                values = tryparse.(Float64, fields)
                any(isnothing, values) &&
                    throw(ArgumentError("Non-numeric atomic-configuration row for $current_element at POTCAR line $line_index"))
                table[row, :] = values
            end

            haskey(configurations, current_element) &&
                throw(ArgumentError("Multiple atomic configurations found for $current_element in POTCAR"))
            configurations[current_element] = table
        end
    end

    result = Dict{String, Matrix{Float64}}()
    for element in elements
        haskey(configurations, element) ||
            throw(ArgumentError("No atomic configuration for POSCAR element $element in POTCAR"))
        haskey(zvals, element) ||
            throw(ArgumentError("No ZVAL value for POSCAR element $element in POTCAR"))
        result[element] = configurations[element]
    end

    return result, Dict{String, Float64}(element => zvals[element] for element in elements)
end

"""
    dft_orbitals(configurations, zvals)

Return `(n, l, occupancy)` tuples for the subshells included in each element's
DFT valence configuration. Occupied configuration rows are considered from
highest to lowest energy until their occupations sum to the element's `ZVAL`;
occupancies from rows with the same `(n, l)` are combined.
"""
function dft_subshells(
    configurations::AbstractDict{<:AbstractString, <:AbstractMatrix},
    zvals::AbstractDict{<:AbstractString, <:Real},
)
    dft_subshells_occupancy = Dict{String, Vector{Tuple{Int, Int, Float64}}}()

    for (element, zval) in zvals
        haskey(configurations, element) ||
            throw(ArgumentError("No atomic configuration provided for $element"))
        isfinite(zval) && zval >= 0 ||
            throw(ArgumentError("ZVAL for $element must be a finite, non-negative number"))

        configuration = configurations[element]
        size(configuration, 2) == 5 ||
            throw(ArgumentError("Atomic configuration for $element must have five columns"))

        rows = sortperm(configuration[:, 4]; rev=true)
        electrons = 0.0
        shell_occupancies = Dict{Tuple{Int, Int}, Float64}()

        for row in rows
            occupation = configuration[row, 5]
            occupation > 0 || continue

            n_value, l_value = configuration[row, 1:2]
            isinteger(n_value) && isinteger(l_value) ||
                throw(ArgumentError("Non-integer n or l value in atomic configuration for $element"))
            shell = (Int(n_value), Int(l_value))
            shell_occupancies[shell] = get(shell_occupancies, shell, 0.0) + occupation
            electrons += occupation

            if electrons + 1e-8 >= zval
                break
            end
        end

        electrons + 1e-8 >= zval ||
            throw(ArgumentError("Atomic configuration for $element has fewer occupied electrons than ZVAL"))
        dft_subshells_occupancy[String(element)] = [
            (n, l, shell_occupancies[(n, l)])
            for (n, l) in sort!(collect(keys(shell_occupancies)))
        ]
    end

    return dft_subshells_occupancy
end

"""
    total_electrons(zvals, atom_counts)

Calculate the total valence-electron count across the structure using each
element's POTCAR `ZVAL`. `atom_counts` maps element symbols to their number of
atoms.
"""
function total_electrons(
    zvals::AbstractDict{<:AbstractString, <:Real},
    atom_counts::AbstractDict{<:AbstractString, <:Integer},
)
    total = 0.0

    for (element, zval) in zvals
        haskey(atom_counts, element) ||
            throw(ArgumentError("No atom count provided for $element"))
        isfinite(zval) && zval >= 0 ||
            throw(ArgumentError("ZVAL for $element must be a finite, non-negative number"))
        count = atom_counts[element]
        count >= 0 ||
            throw(ArgumentError("Atom count for $element must be non-negative"))

        total += count * zval
    end

    for element in keys(atom_counts)
        haskey(zvals, element) ||
            throw(ArgumentError("No ZVAL value provided for $element"))
    end

    return total
end

function total_electrons(
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
    isempty(elements) &&
        throw(ArgumentError("POSCAR element list is empty"))
    length(elements) == length(counts) ||
        throw(ArgumentError("POSCAR element and atom-count lists have different lengths"))

    atom_counts = Dict{String, Int}(elements[i] => counts[i] for i in eachindex(elements))
    _, zvals = read_POTCAR(poscar_path, potcar_path)

    return total_electrons(zvals, atom_counts)
end