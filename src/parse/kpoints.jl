"""
    read_kpoints(file::AbstractString) -> Matrix{Float64}

Read the k-points of a VASP `KPOINTS` file, in units of the reciprocal lattice vectors (3 × Nk), such that the same file
can be used for structures with different lattices. Supported formats:

- explicit list (`N > 0` k-points in reciprocal coordinates; the weights are ignored),
- line mode (`Line-mode`, reciprocal coordinates): `N` points per segment, including both end points (as in VASP),
- automatic grids (`N = 0`, `Gamma` or `Monkhorst-Pack` with the subdivisions and an optional shift): the full grid,
  without symmetry reduction.

Cartesian coordinates and fully automatic grids are not supported, since they depend on the lattice.
"""
function read_kpoints(file::AbstractString)
    # Labels and comments after '!' (e.g. high-symmetry points in line mode) are ignored
    lines = [split_line(first(split(line, '!'))) for line in open_and_read(file)]
    length(lines) ≥ 4 || error("KPOINTS file $file is too short.")
    N = parse(Int64, lines[2][1])
    mode = lowercase(lines[3][1])
    cartesian(line) = !isempty(line) && lowercase(line[1])[1] in ('c', 'k')
    coords(line) = parse.(Float64, line[1:3])

    if startswith(mode, "l")
        cartesian(lines[4]) && error("KPOINTS file $file: Cartesian coordinates are not supported, use reciprocal coordinates.")
        N ≥ 2 || error("KPOINTS file $file: line mode needs at least 2 points per segment.")
        points = [coords(line) for line in lines[5:end] if length(line) ≥ 3]
        iseven(length(points)) || error("KPOINTS file $file: line mode needs pairs of start and end points.")
        segments = [[k1 .+ (k2 .- k1) .* (n / (N - 1)) for n in 0:N-1] for (k1, k2) in zip(points[1:2:end], points[2:2:end])]
        return reduce(hcat, reduce(vcat, segments))
    elseif N > 0
        cartesian(lines[3]) && error("KPOINTS file $file: Cartesian coordinates are not supported, use reciprocal coordinates.")
        length(lines) ≥ 3 + N || error("KPOINTS file $file: expected $N k-points.")
        return reduce(hcat, [coords(lines[3+n]) for n in 1:N])
    elseif mode[1] in ('g', 'm')
        subdivisions = parse.(Int64, lines[4][1:3])
        shift = length(lines) ≥ 5 && length(lines[5]) ≥ 3 ? coords(lines[5]) : zeros(3)
        # Gamma: (i + s) / q; Monkhorst-Pack: (2r - q - 1) / 2q + s / q with r = i + 1
        offset = mode[1] == 'g' ? zeros(3) : (1 .- subdivisions) ./ (2 .* subdivisions)
        grid = Iterators.product((0:q-1 for q in subdivisions)...)
        return reduce(hcat, [(collect(i) .+ shift) ./ subdivisions .+ offset for i in grid])
    else
        error("KPOINTS file $file: unsupported format (supported: explicit list, line mode, Gamma and Monkhorst-Pack grids).")
    end
end
