"""
    read_xdatcar(xdatcar::AbstractString) -> Xdatcar

Read the configurations in the `xdatcar` file and store them in an `Xdatcar` object.

# Arguments
- `xdatcar::AbstractString`: The path to the XDATCAR file.

# Returns
- `lattice::Array{Float64, 3}`: A 3x3 array representing the lattice vectors.
- `configs::Array{Float64, 3}`: A 3D array of shape (3, Nion, Nconfig), where each 3xNion slice represents the atomic positions in a configuration.

"""
function read_xdatcar(xdatcar="XDATCAR"; frac=true)
    lines = open_and_read(xdatcar)
    lines = split_lines(lines)

    # Scaling parameter
    a = parse(Float64, lines[2][1])

    # Lattice vectors
    lattice = zeros(Float64, 3, 3)
    for i in 1:3
        lattice[:, i] = @. a * parse(Float64, lines[2+i])
    end

    # Number of ions
    Nion = sum(parse.(Int64, lines[7]))

    # Find starting line of configurations
    i_start, _ = next_line_with("Direct", lines)

    # Calculate the number of configurations
    L = length(lines)
    Nconfig = Int((L - i_start + 1) / (Nion + 1))

    # Initialize the configurations array
    configs = zeros(Float64, 3, Nion, Nconfig)

    # Parse the configurations
    for j in 1:Nconfig, i in 1:Nion
        k = j + i_start + Nion * (j - 1) + i - 1
        configs[:, i, j] = frac ? parse.(Float64, lines[k][1:3]) : frac_to_cart(parse.(Float64, lines[k][1:3]), lattice)
    end

    return lattice, configs
end