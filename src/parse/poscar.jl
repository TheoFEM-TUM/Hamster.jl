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
    atom_names :: Array{AbstractString, 1}
    atom_numbers :: Array{Int64, 1}
    rs_atom :: Array{Float64, 2}
    atom_types :: Array{String, 1}
end

"""
    read_poscar(poscar::AbstractString) -> Poscar

Extract all data from the POSCAR file at `poscar`.

# Arguments
- `poscar::AbstractString`: The path to the POSCAR file.

# Returns
- `Poscar`: A `Poscar` struct containing all the extracted data from the POSCAR file, including:
    - `a`: The scaling factor.
    - `lattice`: The 3x3 array of lattice vectors.
    - `atom_names`: An array of atom names.
    - `atom_numbers`: An array of the number of each type of atom.
    - `rs_atom`: A 3xNion array of atomic positions.
    - `atom_types`: An array of atom types corresponding to each atom position.
"""
function read_poscar(poscar)
    lines = open_and_read(poscar)
    lines = split_lines(lines)

    # Scaling parameter
    a = parse(Float64, lines[2][1])

    # Lattice vectors
    lattice = zeros(Float64, 3, 3)
    for i in 1:3
        lattice[:, i] = @. a * parse(Float64, lines[2+i])
    end

    # Atom names and numbers
    atom_names = lines[6]
    atom_numbers = parse.(Int64, lines[7])
    if !(length(atom_names) == length(atom_numbers))
        @info "Length of atom_names and atom_numbers not equal!"; end
    atom_types = String[]
    for (k, atom_number) in enumerate(atom_numbers), _ in 1:atom_number
        push!(atom_types, atom_names[k])
    end

    # Atom positions and types
    Nion = sum(atom_numbers)
    rs_atom = zeros(Float64, 3, Nion)
    for i in 1:Nion
       rs_atom[:, i] = [parse(Float64, el) for el in lines[8+i][1:3]]
    end
    return Poscar(a, lattice, atom_names, atom_numbers, rs_atom, atom_types)
end