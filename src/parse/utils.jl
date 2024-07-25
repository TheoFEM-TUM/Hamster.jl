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
function split_lines(lines; char=" ")
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
split_line(line; char=" ") = filter(!isempty, split(line, char))

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
    M = parse.(type, lines[2:end])
    return reshape(M, Ns)
end