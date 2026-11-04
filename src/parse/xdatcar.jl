"""
    read_xdatcar(xdatcar="XDATCAR"; frac=true, verbosity=1) -> (lattice, configs)

Read the configurations in the XDATCAR file `xdatcar` (VASP 5 format).

# Arguments
- `xdatcar::AbstractString`: The path to the XDATCAR file.
- `frac::Bool=true`: If false, coordinates are transformed to cartesian from fractional.
- `verbosity::Int64=1`: If greater 1, progress is printed.

# Returns
- `lattice`: The lattice vectors (columns) as a 3x3 matrix, or as a 3x3xNconfig array if the cell changes between the
  configurations (variable-cell runs, where the header is repeated before every configuration).
- `configs::Array{Float64, 3}`: A 3D array of shape (3, Nion, Nconfig), where each 3xNion slice represents the atomic
  positions in a configuration (fractional or Cartesian, see `frac`). Configurations in the file can be given as
  `Direct` or `Cartesian` coordinates.
"""
function read_xdatcar(xdatcar="XDATCAR"; frac=true, verbosity=1)
    Nconfig = 0
    lattices = zeros(3, 3, 0)
    configs = zeros(3, 0, 0)
    open(xdatcar, "r") do io
        for line in Iterators.reverse(eachline(io))
            if is_configuration_line(line)
                Nconfig = parse(Int64, split(line, [' ', '='], keepempty=false)[end])
                break
            end
        end

        seekstart(io)
        lattice = zeros(3, 3)
        a = 1.0
        Nion = 0
        iconfig = 0
        while !eof(io)
            line = readline(io)
            if is_configuration_line(line)
                iconfig += 1
                verbosity > 1 && println("Config $iconfig / $Nconfig")
                cartesian = lowercase(strip(line))[1] ∈ ('c', 'k')
                for iion in 1:Nion
                    if eof(io)
                        @warn "$xdatcar: configuration $iconfig is incomplete and is skipped."
                        iconfig -= 1
                        configs = configs[:, :, 1:iconfig]
                        lattices = lattices[:, :, 1:iconfig]
                        return
                    end
                    configs[:, iion, iconfig] = parse_numbers(split(readline(io)), 3, xdatcar, "position")
                end
                r = view(configs, :, :, iconfig)
                if cartesian && frac
                    r .= lattice \ (a .* r)
                elseif cartesian
                    r .*= a
                elseif !frac
                    r .= frac_to_cart(r, lattice)
                end
                lattices[:, :, iconfig] = lattice
            elseif !isempty(strip(line))
                a, lattice, Nion = read_xdatcar_header(io, xdatcar)
                if isempty(configs)
                    configs = zeros(3, Nion, Nconfig)
                    lattices = zeros(3, 3, Nconfig)
                end
            end
        end
    end
    lattice = all(n -> lattices[:, :, n] == lattices[:, :, 1], axes(lattices, 3)) ? lattices[:, :, 1] : lattices
    return lattice, configs
end

"""
    is_configuration_line(line)

Whether `line` of an XDATCAR file starts a configuration (`Direct configuration= n`, `Cartesian configuration= n`).
"""
is_configuration_line(line) = occursin("configuration", lowercase(line))

"""
    read_xdatcar_header(io, file) -> (a, lattice, Nion)

Read the header of an XDATCAR file after its first line (scale factor, lattice vectors, element names and numbers of
atoms). Returns the length scale `a`, the scaled lattice vectors (columns) and the number of atoms.
"""
function read_xdatcar_header(io, file)
    scale = parse_numbers(split(readline(io)), 1, file, "scale factor")[1]
    lattice = reduce(hcat, [parse_numbers(split(readline(io)), 3, file, "lattice vector") for _ in 1:3])
    a = length_scale(scale, lattice)
    names = split(readline(io))
    all(isnothing ∘ (s -> tryparse(Int64, s)), names) ||
        error("$file: the header must contain the element names (VASP 4 files without them are not supported).")
    Nion = sum(parse.(Int64, split(readline(io))))
    return a, a .* lattice, Nion
end
