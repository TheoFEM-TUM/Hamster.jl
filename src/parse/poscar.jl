element_to_number(element)::UInt8 = elements[Symbol(element)].number
number_to_element(number) = elements[number].symbol

"""
    struct Poscar

A structure to represent the data contained in a VASP POSCAR file.

# Fields
- `a::Float64`: The scaling factor.
- `lattice::Array{Float64, 2}`: A 3x3 array representing the lattice vectors.
- `atom_names::Array{AbstractString, 1}`: An array of atom names.
- `atom_numbers::Array{Int64, 1}`: An array of the number of each type of atom.
- `rs_atom::Array{Float64, 2}`: A 3xNion array of atomic positions.
- `atom_types::Array{String, 1}`: An array of atom types corresponding to each atom position.
"""
struct Poscar
    a :: Float64
    lattice :: Array{Float64, 2}
    atom_names :: Vector{String}
    atom_numbers :: Array{Int64, 1}
    rs_atom :: Array{Float64, 2}
    atom_types :: Vector{UInt8}
end

"""
    read_poscar(poscar::AbstractString) -> Poscar

Read a POSCAR file in the VASP 5 format (with a line of element names). The positions are returned as fractional
coordinates (`Direct` or `Cartesian` in the file), a `Selective dynamics` line and the flags after the coordinates are
skipped, a negative scale factor is the volume of the cell, and element names with POTCAR suffixes (`Cs_sv`,
`Pb_d/...`) are reduced to the element.
"""
function read_poscar(poscar)
    lines = split_lines(open_and_read(poscar))
    lattice = reduce(hcat, [parse_numbers(lines[2+i], 3, poscar, "lattice vector") for i in 1:3])
    a = length_scale(parse_numbers(lines[2], 1, poscar, "scale factor")[1], lattice)
    lattice = a .* lattice

    all(isnothing ∘ (s -> tryparse(Int64, s)), lines[6]) ||
        error("$poscar: line 6 must contain the element names (VASP 4 POSCAR files without them are not supported).")
    atom_names = poscar_element.(lines[6], Ref(poscar))
    atom_numbers = parse.(Int64, lines[7][1:length(atom_names)])

    atom_types = UInt8[]
    for (k, atom_number) in enumerate(atom_numbers), _ in 1:atom_number
        push!(atom_types, element_to_number(atom_names[k]))
    end

    iline = 8
    startswith(lowercase(first(lines[iline])), "s") && (iline += 1)
    cartesian = lowercase(first(lines[iline]))[1] ∈ ('c', 'k')

    Nion = sum(atom_numbers)
    rs_atom = zeros(Float64, 3, Nion)
    for i in 1:Nion
        rs_atom[:, i] = parse_numbers(lines[iline+i], 3, poscar, "position")
    end
    if cartesian
        rs_atom = lattice \ (a .* rs_atom)
    end
    return Poscar(a, lattice, atom_names, atom_numbers, rs_atom, atom_types)
end

"""
    parse_numbers(fields, n, file, what) -> Vector{Float64}

The first `n` fields of a line as numbers; an error naming `file` and `what` if there are fewer or they are no numbers.
"""
function parse_numbers(fields, n, file, what)
    numbers = length(fields) ≥ n ? tryparse.(Float64, fields[1:n]) : [nothing]
    any(isnothing, numbers) && error("$file: could not read the $what from the line \"$(join(fields, ' '))\".")
    return Float64.(numbers)
end

"""
    length_scale(scale, lattice)

Factor for the lattice vectors (columns of `lattice`, as in the file) and Cartesian positions of a POSCAR or XDATCAR
file with scale factor `scale`; a negative `scale` is the volume of the cell.
"""
length_scale(scale, lattice) = scale < 0 ? cbrt(abs(scale) / abs(det(lattice))) : scale

"""
    poscar_element(name, file) -> String

Element symbol of an element name in a POSCAR file, without POTCAR suffixes (`Cs_sv` -> `Cs`, `Pb_d/a1b2` -> `Pb`).
"""
function poscar_element(name, file)
    symbol = first(split(name, ['_', '/']))
    haskey(elements, Symbol(symbol)) || error("$file: unknown element \"$name\".")
    return String(symbol)
end
