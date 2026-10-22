"""
    MDData

Reference data of the MD fit: `positions` (3 × N × Nf, Cartesian), `lattice` (3 × 3 × Nf), atom `types`, total
`energies` (eV, Nf) and `forces` (eV/Å, 3 × N × Nf), and the `source` (file and frame) of every frame.
"""
struct MDData
    positions :: Array{Float64, 3}
    lattice :: Array{Float64, 3}
    types :: Vector{UInt8}
    energies :: Vector{Float64}
    forces :: Array{Float64, 3}
    source :: Vector{String}
end

nframes(data::MDData) = length(data.energies)

"""
    read_vasprun(filename; energy="e_fr_energy")

Frames of a VASP run (e.g. an MD with IBRION = 0) from `vasprun.xml`: lattice, positions, forces and the energy
`energy` (default: the free energy, whose derivative the VASP forces are) of every ionic step.
"""
function read_vasprun(filename; energy="e_fr_energy")
    types = UInt8[]
    lattices = Matrix{Float64}[]; positions = Matrix{Float64}[]; forces = Matrix{Float64}[]; energies = Float64[]
    lattice = zeros(3, 3)
    pos = zeros(3, 0)
    frc = zeros(3, 0)
    e_last = NaN
    in_calc = false; in_atoms = false; in_scstep = false
    mode = :none; row = 0
    for line in eachline(filename)
        s = strip(line)
        if startswith(s, "<array name=\"atoms\"")
            in_atoms = true
        elseif in_atoms && startswith(s, "</array>")
            in_atoms = false
        elseif in_atoms && startswith(s, "<rc><c>")
            el = strip(split(split(s, "<c>")[2], "</c>")[1])
            push!(types, element_to_number(el))
        elseif startswith(s, "<calculation>")
            in_calc = true; e_last = NaN
        elseif startswith(s, "</calculation>")
            if in_calc && size(pos, 2) == length(types) && size(frc, 2) == length(types)
                push!(lattices, copy(lattice)); push!(positions, lattice * pos); push!(forces, copy(frc)); push!(energies, e_last)
            end
            in_calc = false
        elseif startswith(s, "<scstep>")
            in_scstep = true
        elseif startswith(s, "</scstep>")
            in_scstep = false
        elseif in_calc && startswith(s, "<varray name=\"basis\"")
            mode = :basis; row = 0
        elseif in_calc && startswith(s, "<varray name=\"positions\"")
            mode = :positions; row = 0; pos = zeros(3, length(types))
        elseif in_calc && startswith(s, "<varray name=\"forces\"")
            mode = :forces; row = 0; frc = zeros(3, length(types))
        elseif startswith(s, "</varray>")
            mode = :none
        elseif mode ≠ :none && startswith(s, "<v>")
            row += 1
            v = parse.(Float64, split(replace(s, "<v>" => "", "</v>" => "")))
            if mode == :basis
                lattice[:, row] = v
            elseif mode == :positions
                pos[:, row] = v
            else
                frc[:, row] = v
            end
        elseif in_calc && !in_scstep && occursin("name=\"$energy\"", s)
            e_last = parse(Float64, replace(split(s, ">")[2], "</i" => ""))
        end
    end
    Nf = length(energies)
    return MDData(stack(positions), stack(lattices), types, energies, stack(forces), ["$filename:$n" for n in 1:Nf])
end

"""
    read_md_data(filename)

Reference data from a `vasprun.xml` or an HDF5 file with the datasets `positions` (3 × N × Nf, Cartesian, Å),
`lattice` (3 × 3 or 3 × 3 × Nf), `types` (proton numbers), `energies` (eV) and `forces` (eV/Å), as written by
`write_md_data`.
"""
function read_md_data(filename)
    if endswith(lowercase(filename), ".xml")
        return read_vasprun(filename)
    end
    h5open(filename, "r") do f
        pos = read(f["positions"])
        lat = read(f["lattice"])
        lat = ndims(lat) == 2 ? repeat(lat, 1, 1, size(pos, 3)) : lat
        types = UInt8.(read(f["types"]))
        return MDData(pos, lat, types, read(f["energies"]), read(f["forces"]), ["$filename:$n" for n in axes(pos, 3)])
    end
end

"""
    write_md_data(filename, data)

Write reference data in the HDF5 format of `read_md_data`.
"""
function write_md_data(filename, data::MDData)
    h5open(filename, "w") do f
        f["positions"] = data.positions; f["lattice"] = data.lattice; f["types"] = Int.(data.types)
        f["energies"] = data.energies; f["forces"] = data.forces
    end
end

"""
    load_md_data(files, conf; stride=1, skip=get_md_skip(conf))

Frames of all `files`, skipping the first `skip` frames of each file and taking every `stride`-th frame.
"""
function load_md_data(files, conf=get_empty_config(); stride=1, skip=get_md_skip(conf))
    isempty(files) && return nothing
    sets = map(files) do file
        d = read_md_data(file)
        sel = (skip + 1):stride:nframes(d)
        MDData(d.positions[:, :, sel], d.lattice[:, :, sel], d.types, d.energies[sel], d.forces[:, :, sel], d.source[sel])
    end
    all(s -> s.types == sets[1].types, sets) || error("The MD data files have different atoms (types or order).")
    return MDData(cat((s.positions for s in sets)...; dims=3), cat((s.lattice for s in sets)...; dims=3), sets[1].types,
                  reduce(vcat, (s.energies for s in sets)), cat((s.forces for s in sets)...; dims=3),
                  reduce(vcat, (s.source for s in sets)))
end
