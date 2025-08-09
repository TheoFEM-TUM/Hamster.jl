"""
    read_xdatcar(xdatcar::AbstractString) -> Xdatcar

Read the configurations in the `xdatcar` file and store them in an `Xdatcar` object.

# Arguments
- `xdatcar::AbstractString`: The path to the XDATCAR file.

# Returns
- `lattice::Array{Float64, 3}`: A 3x3 array representing the lattice vectors.
- `configs::Array{Float64, 3}`: A 3D array of shape (3, Nion, Nconfig), where each 3xNion slice represents the atomic positions in a configuration.

"""
function read_xdatcar_stream(xdatcar="XDATCAR"; frac=true)
    Nconfig = 0
    Nion = 0
    a = 0.
    lattice = zeros(3, 3)
    configs = zeros(3, 1, 1)
    open(xdatcar, "r") do io
        # Read total number of configs from end of file
        for line in Iterators.reverse(eachline(io))
            if occursin("Direct", line)
                Nconfig = parse.(Int64, split(line, ' ', keepempty=false)[end])
                break
            end
        end

        # Read number of ions, lattice vectors and lattice constant from start
        seekstart(io)
        for (i, line) in enumerate(eachline(io))
            if i == 2
                a = parse(Int64, split(line, ' ', keepempty=false)[1])
            elseif i ∈ [3, 4, 5]
                lattice[:, i-2] = a .* parse.(Float64, split(line, ' ', keepempty=false))
            elseif i == 7
                Nion = sum(parse.(Int64, split(line, ' ', keepempty=false)))
                break
            end
        end

        # Read ion positions
        configs = zeros(3, Nion, Nconfig)
        found_configs = 0
        found_ions = 0
        ion_block = false
        for line in eachline(io)
            if occursin("Direct", line)
                ion_block = true
                found_configs += 1
                @show found_configs
            elseif ion_block
                found_ions += 1
                eachsplit
                @views for (i, num) in enumerate(eachsplit(line, ' ', keepempty=false))
                    configs[i, found_ions, found_configs] = parse(Float64, num)
                end
                if !frac
                    configs[:, found_ions, found_configs] .= frac_to_cart(configs[:, found_ions, found_configs], lattice)
                end
            end
            if found_ions == Nion
                ion_block = false
                found_ions = 0
            end
        end
    end
    return lattice, configs
end