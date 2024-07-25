const COMMENT_SYMBOLS = ['!', '#', '%']

"""
    get_config(; filename="hconf")

Retrieves the configuration from a file if it exists, or returns an empty configuration if the file does not exist.

# Arguments
- `filename` (optional): The name of the configuration file to check for existence. Defaults to `"hconf"`.

# Returns
- `Config`: A `Config` instance.
"""
get_config(;filename="hconf")::Config = read_config(filename)

"""
    read_config()

Read the tbconfig file if given, else return default config.
"""
function read_config(file::String)
    conf = get_empty_config()
    if isfile(file)
        lines = open_and_read(file)
        #lines = split_lines(lines)
        blocks = split_blocks(lines)
        for block in blocks
            read_config_block!(block, conf)
        end
    else
        @info "File $file not found. Using default values."
    end
    return conf
end

"""
    ConfigBlock

Block with options in config file enclosed by begin and end statement.
"""
struct ConfigBlock
    header :: Vector{String}
    content :: Vector{String}
end

"""
    read_config_block(block; config=Dict())

Read a `block` in the tbconfig file and store all options to the config
dictionary.
"""
function read_config_block!(block::ConfigBlock, conf::Config)
    for line in block.content
        if occursin('=', line) && length(filter_comment(line)) > 0
            option_key, option_value = read_config_line(filter_comment(line))
            if block.header[end] == "Options"
                set_value!(option_key, option_value, conf)
            else
                set_value!(option_key, block.header[end], option_value, conf)
            end
        end
    end
end

"""
    split_blocks(lines)

Splits a list of lines into blocks of text based on "begin" and "end" markers. Each block is encapsulated in a `ConfigBlock` struct.

# Arguments
- `lines`: A list of strings, where each string is a line from the input text.

# Returns
- A list of `ConfigBlock` instances, each containing the header and content of a block found in the input lines.
"""
function split_blocks(lines)
    blocks = ConfigBlock[]
    found_begin = false; found_end = false
    begin_line = 0; end_line = 0; unclosed_begins = 0
    for (k, line) in enumerate(lines)
        if "begin" in split_line(line)
            if unclosed_begins == 0
                begin_line = k
                found_begin = true
            end
            unclosed_begins += 1
        elseif "end" in split_line(line)
            if unclosed_begins == 1
                end_line = k
                found_end = true
                unclosed_begins = 0
            else
                unclosed_begins -= 1
            end
        end
        if found_begin && found_end
            block_content = lines[begin_line+1:end_line-1]
            block_header = string.([el for el in split_line(lines[begin_line]) if el ≠ "begin"])
            block = ConfigBlock(block_header, block_content)
            push!(blocks, block)
            found_begin = false; found_end = false
        end
    end
    return blocks
end

"""
    read_config_line(line)

Read a `line` of the tbconfig file and extract `option_key` and `option_value`.
"""
function read_config_line(line)
    option_key, option_value = split_line(line, char="=")
    option_key = strip(option_key, ' ')
    option_value = strip(option_value, ' ')
    return option_key, option_value
end

function parse_line(type, line)
    lineparse = tryparse.(type, lowercase.(line))
    if all(x->x≠nothing, lineparse)
        if length(lineparse) == 1; return lineparse[1]; end
    end
    return lineparse
end

"""
    filter_comment(line)

If `line` contains a comment, return the non-commented part of `line`.
"""
function filter_comment(line)
    for comment_symbol in COMMENT_SYMBOLS
        if occursin(comment_symbol, line)
            line_no_comment = strip(split_line(line, char=comment_symbol)[1], ' ')
            if length(strip(line_no_comment, ' ')) > 0 
                return line_no_comment
            else
                return ""
            end
        end
    end
    return line
end

"""
    check_parse(type, line)

Check if the string or vector of strings `line` can be converte to `type`.
"""
function check_parse(type, line)
    lineparse = parse_line(type, line)
    return all(x->x≠nothing, lineparse)
end

"""
    convert_line(line)

Convert `line` if possible to different type.
"""
function convert_line(line)
    for type in [Int64, Float64, Bool]
        if check_parse(type, line)
            return parse_line(type, line)
        end
    end
    if typeof(line) <: AbstractArray && length(line) == 1
        return line[1]; else return line; end
end






