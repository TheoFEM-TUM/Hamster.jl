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
    TEMP_DIR

Folder for the temporary files of a run (per-rank files, eigenvalues of single structures), relative to the run
folder. `main` sets it to a new folder `hamster_tmp_<random>` for every run and removes that folder at the end;
outside of `main` it is `tmp`.
"""
const TEMP_DIR = Ref("tmp")

"""
    temp_path(parts...) -> String

Path of a file or folder in the temporary folder of the run (see [`TEMP_DIR`](@ref Hamster.TEMP_DIR)).
"""
temp_path(parts...) = joinpath(TEMP_DIR[], parts...)

"""
    make_temp_dir(parts...) -> String

Create the folder `temp_path(parts...)` (and its parents) if it does not exist and return its path. Several ranks
may call this at the same time.
"""
function make_temp_dir(parts...)
    path = temp_path(parts...)
    try
        mkpath(path)
    catch
        isdir(path) || rethrow()
    end
    return path
end

"""
    get_rank_filename(filename::AbstractString, rank::Integer; temp=true) -> String

Return the per-rank file name `<basename>_<rank>.<ext>`, in the temporary folder of the run if `temp = true`.
Example:
    get_rank_filename("data.h5", 3) == "tmp/data_3.h5"   # outside of `main`
"""
function get_rank_filename(filename::AbstractString, rank::Integer; temp=true)
    base = splitext(basename(filename))[1]
    ext  = splitext(filename)[2]
    if temp
        return temp_path(@sprintf("%s_%d%s", base, rank, ext))
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
    collapse_files_with(str; location=temp_path(), outname=str)

Read the files `<str><index>.dat` in the folder `location`, combine their arrays in the order of `index` along a new
last dimension and write the result to `<outname>.dat`. Returns `false` (and writes nothing) if there are no such
files.
"""
function collapse_files_with(str; location=temp_path(), outname=str)
    isdir(location) || return false
    pattern = Regex("^" * str * raw"(\d+)\.dat$")
    matches = filter(!isnothing, [match(pattern, file) for file in readdir(location)])
    isempty(matches) && return false
    indices = [parse(Int64, m[1]) for m in matches]
    data = [read_from_file(joinpath(location, m.match)) for m in matches[sortperm(indices)]]
    ndims = length(size(data[1]))
    data_combined = cat(data..., dims=ndims+1)
    write_to_file(data_combined, outname)
    return true
end