"""
    read_eigenval(file::AbstractString, nmax::Int=1000000)

Read the `EIGENVAL` VASP output file and extract the k-points, energy bands and occupancies.

# Arguments
- `file::AbstractString`: The path to the `EIGENVAL` file.
- `nmax::Int=1000000`: The maximum number of bands to read. Default is 1,000,000.

# Returns
- `kpoints::Array{Float64, 2}`: A 3 x nkpts array where each column represents a k-point.
- `E_bands::Array{Float64, 2}`: An nbands x nkpts array of energy bands.
- `occs::Array{Float64, 2}`: An nbrand × nkpts array of band occupancies.
"""
function read_eigenval(file::AbstractString, nmax::Int=1000000)
    lines = open_and_read(file)
    lines = split_lines(lines)
    meta = Dict{String, Float64}()
    @views begin
        nelec = parse(Int64, lines[6][1]); meta["nelec"] = nelec
        nkpts = parse(Int64, lines[6][2]); meta["nkpts"] = nkpts
        nbands = parse(Int64, lines[6][3]); meta["nbands"] = nbands
        meta["temp"] = parse(Float64, lines[3][1])
        nbands = min(nmax, nbands)
        empty_lines = [i for (i, line) in enumerate(lines) if isempty(line)]
        E_bands = zeros(nbands, nkpts)
        kpoints = zeros(3, nkpts)
        occs = zeros(nbands, nkpts)
        for (k, i) in enumerate(empty_lines)
            kpoints[:, k] = parse.(Float64, lines[i+1][1:3])
            for j in 1:nbands
                @inbounds E_bands[j, k] = parse(Float64, lines[i+1+j][2])
                if length(lines[i+1+j]) ≥ 3
                    @inbounds occs[j, k] = parse(Float64, lines[i+1+j][3])
                end
            end
        end
    end
    return kpoints, E_bands, occs
end