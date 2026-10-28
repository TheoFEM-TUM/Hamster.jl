"""
    print_hamster(io_stream=stdout)

Prints an ASCII art Hamster logo and description.

# Arguments
- `io_stream::IO`: The output stream to which the text should be printed. Defaults to `stdout`.
"""
function print_hamster(io_stream=stdout)
    println(io_stream, """

    Welcome to
    ===========================================================================
    ||   _   _                             _                         _   _   ||
    ||  | | | |   __ _   _ __ ___    ___  | |_    ___   _ __        (_) | |  ||
    ||  | |_| |  / _` | | '_ ` _ \\  / __| | __|  / _ \\ | '__|       | | | |  ||
    ||  |  _  | | (_| | | | | | | | \\__ \\ | |_  |  __/ | |     _    | | | |  ||
    ||  |_| |_|  \\__,_| |_| |_| |_| |___/  \\__|  \\___| |_|    (_)  _/ | |_|  ||
    ||                                                            |__/       ||
    ===========================================================================
    Hamiltonian-learning
        Approach for Multiscale Simulations 
            using a Transferable and Efficient Representation.
    
    """)
end

"""
    open_and_read(file::AbstractString) -> Vector{String}

Open a file, read all lines, and return them as a vector of strings.

# Arguments
- `file::AbstractString`: The path to the file to be read.

# Returns
- `lines::Vector{String}`: A vector where each element is a line from the file.
"""
function open_and_read(file)
    f = open(file)
    lines = readlines(f)
    close(f)
    return lines
end

"""
    split_lines(lines::Vector{String}) -> Vector{Vector{String}}

Split each line of the input vector of strings into its constituent non-empty elements.

# Arguments
- `lines::Vector{String}`: A vector of strings, where each string represents a line to be split.

# Returns
- `split_lines::Vector{Vector{String}}`: A vector of vectors of strings, where each inner vector 
contains the non-empty elements of the corresponding line from the input.
"""
function split_lines(lines; char=r"\s+")
    split_lines = Vector{Vector{String}}(undef, length(lines))
    Threads.@threads for l in eachindex(lines)
        split_lines[l] = split_line(lines[l]; char=char)
    end
    return split_lines
end

"""
    split_line(line::String) -> Vector{String}

Splits a line of text into individual words, removing any extra spaces.

# Arguments
- `line::String`: A string representing the line of text to be split.

# Returns
- `Vector{String}`: An array of words from the input line, excluding any empty elements.
"""
split_line(line; char=r"\s+") = filter(!isempty, split(line, char))

"""
    parse_lines_as_array(lines; i1=1, i2=3, type=Float64)

Parses a list of strings into a 2D array, extracting and converting elements from each line based on specified indices and type.

# Arguments
- `lines`: A list of strings, where each string is a line containing elements to be parsed.
- `i1` (optional): The starting index for the elements to extract from each line. Defaults to `1`.
- `i2` (optional): The ending index for the elements to extract from each line. Defaults to `3`.
- `type` (optional): The type to which the extracted elements should be converted. Defaults to `Float64`.

# Returns
- A 2D array of the specified `type`, where each row corresponds to a line and each column corresponds to an extracted element from the line.
"""
function parse_lines_as_array(lines; i1=1, i2=3, type=Float64)
    Nj = length(lines); Ni = length(i1:i2)
    array = Array{type, 2}(undef, Nj, Ni)
    for (j, line) in enumerate(lines)
        array[j, :] = parse.(type, line[i1:i2])
    end
    return array
end

"""
    next_line_with(keywords, lines)
    
Find the next line in lines that contains a set of keywords.
"""
function next_line_with(keywords::AbstractArray, lines)
    index = false; keyline = false
    for (i, line) in enumerate(lines)
        if all(keyword->keyword in line, keywords)
            index = i
            keyline = line
            return index, keyline
        end
    end
end

next_line_with(s::String, lines) = next_line_with([s], lines)

"""
    get_rank_filename(filename::AbstractString, rank::Integer) -> String

Return a modified file path of the form `tmp/<basename>_<rank>.<ext>`.
Example:
    rank_filename("data.h5", 3) == "tmp/data_3.h5"
"""
function get_rank_filename(filename::AbstractString, rank::Integer; temp=true)
    base = splitext(basename(filename))[1]
    ext  = splitext(filename)[2]
    if temp
        return joinpath("tmp", @sprintf("%s_%d%s", base, rank, ext))
    else
        return @sprintf("%s_%d%s", base, rank, ext)
    end
end

"""
    write_to_file(M, filename)

Write the Array `M` and shape to a new file with name `filename`.
"""
function write_to_file(M, filename)
    file = open(filename*".dat", "w")
    print(file, "   "); for k in 1:length(size(M)); print(file, size(M, k), "  "); end; print(file, "\n")
    for e in collect(Iterators.flatten(M))
        println(file, e)
    end
    close(file)
end

"""
    read_from_file(filename, type)

Read an Array `M` with elements of `type` from the file with name `filename`.
"""
function read_from_file(filename; type=Float64)
    lines = open_and_read(filename)
    Ns = Tuple(parse.(Int64, filter!(el->el≠"", split(lines[1], " "))))
    M = tryparse.(type, lines[2:end])
    if any(x->x===nothing, M)
        M = tryparse.(ComplexF64, lines[2:end])
    end
    return reshape(M, Ns)
end

"""
    stack_output_files(str; location=joinpath(pwd(), "tmp")) -> Dict{String, Tuple{Vector{Int64}, Array}}

Collects all output files `<str>_<system>_<config_index>.dat` in `location` and stacks the data of each system along a new last axis, sorted by configuration index.

# Returns
- A dictionary mapping each system name to `(indices, data)`, where `data[..., i]` corresponds to configuration `indices[i]`.
"""
function stack_output_files(str; location=joinpath(pwd(), "tmp"))
    files = [file for file in readdir(location) if startswith(file, "$(str)_")]

    # Group files by everything between `str` and the final index
    groups = Dict{String, Vector{String}}()
    for file in files
        parts = split(replace(file, ".dat" => ""), "_")
        middle = join(parts[2:end-1], "_")
        push!(get!(groups, middle, String[]), file)
    end

    stacked = Dict{String, Tuple{Vector{Int64}, Array}}()
    for (middle, group_files) in groups
        # Extract and sort by trailing index
        indices = [parse(Int, split(replace(file, ".dat" => ""), "_")[end]) for file in group_files]
        p = sortperm(indices)

        # Read and concatenate data
        data = [read_from_file(joinpath(location, file)) for file in group_files[p]]
        stacked[middle] = (indices[p], cat(data..., dims=ndims(data[1]) + 1))
    end
    return stacked
end

"""
    collapse_files_with(str; location=joinpath(pwd(), "tmp"), index_file="eval_config_inds.h5")

Stacks the output files `<str>_<system>_<config_index>.dat` in `location` per system (see `stack_output_files`) and writes the result
to `<str>_<system>.dat` (or `<str>.dat` if the system name is empty).

The sorted configuration indices are written to `index_file` with one dataset per system (`config_inds` if the system name is empty),
such that `data[..., i]` corresponds to configuration `indices[i]`.

# Arguments
- `str::String`: The prefix of the output files (e.g., `"Es"` or `"vs"`).
- `location`: Directory containing the output files.
- `index_file`: HDF5 file the configuration indices are written to.
"""
function collapse_files_with(str; location=joinpath(pwd(), "tmp"), index_file="eval_config_inds.h5")
    h5open(index_file, "w") do h5
        for (middle, (indices, data)) in stack_output_files(str, location=location)
            write_to_file(data, isempty(middle) ? str : "$(str)_$(middle)")
            h5[isempty(middle) ? "config_inds" : middle] = indices
        end
    end
end

const OUTPUT_DATASETS = Dict("Es" => "eigenvalues", "vs" => "eigenvectors")

"""
    collect_output_files(strs; location=joinpath(pwd(), "tmp"), index_file="eval_config_inds.h5", h5_file="eval_eigenval.h5")

Writes the final output files from the per-structure output files `<str>_<system>_<config_index>.dat` in `location` for each prefix in `strs`.

- Named systems (e.g., from a universal `.h5` structure file): the data of all systems is written to `h5_file`, with one group per system containing
  the stacked data (`eigenvalues` for `"Es"`, `eigenvectors` for `"vs"`, with configurations along the last axis) and the corresponding `config_inds`.
- A single unnamed system: a single output file is copied to `<str>.dat`. Multiple output files are stacked with `collapse_files_with`,
  and their configuration indices are written to `index_file`.
"""
function collect_output_files(strs; location=joinpath(pwd(), "tmp"), index_file="eval_config_inds.h5", h5_file="eval_eigenval.h5")
    h5_mode = "w"
    for str in strs
        files = [file for file in readdir(location) if startswith(file, "$(str)_")]
        isempty(files) && continue
        stacked = stack_output_files(str, location=location)
        if all(isempty, keys(stacked))
            if length(files) == 1
                cp(joinpath(location, only(files)), "$str.dat"; force=true)
            else
                collapse_files_with(str, location=location, index_file=index_file)
            end
        else
            h5open(h5_file, h5_mode) do h5
                for (system, (indices, data)) in stacked
                    g = haskey(h5, system) ? h5[system] : create_group(h5, system)
                    g[get(OUTPUT_DATASETS, str, str)] = data
                    if !haskey(g, "config_inds"); g["config_inds"] = indices; end
                end
            end
            h5_mode = "cw"
        end
    end
end