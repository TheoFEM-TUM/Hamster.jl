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
    ConfigBlock

A structure representing a block of configuration settings.

The `ConfigBlock` struct is used to encapsulate a segment of a configuration file, typically consisting of a header and associated content lines. The header usually indicates the section or category of the configuration, while the content contains key-value pairs or other relevant settings.

# Fields:
- `header::Vector{String}`: A vector of strings representing the header of the configuration block. This typically identifies the section or category the block belongs to.
- `content::Vector{String}`: A vector of strings representing the content of the configuration block. Each string usually corresponds to a line in the configuration file, often containing key-value pairs or other settings.
"""
struct ConfigBlock
    header :: Vector{String}
    content :: Vector{String}
end

"""
    read_config_block!(block::ConfigBlock, conf::Config)

Processes a configuration block and updates the configuration object with the parsed key-value pairs.

This function iterates over the lines in a `ConfigBlock`, parses each line to extract configuration options and their values, and updates the provided `Config` object. The function handles lines containing the `=` character, while ignoring comments and empty lines. The updates are made based on whether the block's header ends with `"Options"` or some other string.

# Arguments:
- `block::ConfigBlock`: An object representing a block of configuration lines, typically containing a header and a list of content lines.
- `conf::Config`: The configuration object that will be updated with the parsed key-value pairs from the block.
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
    read_config_line(line::String) -> Tuple{String, String}

Parses a configuration line into a key-value pair.

This function splits a configuration line at the first occurrence of the `=` character, separating the line into an option key and its corresponding value. Both the key and value are stripped of any leading or trailing whitespace.

# Arguments:
- `line::String`: A string representing a single line from a configuration file, typically in the form `"key = value"`.

# Returns:
- A tuple `(option_key, option_value)` where:
  - `option_key::String`: The configuration option key, stripped of leading and trailing whitespace.
  - `option_value::String`: The configuration option value, also stripped of leading and trailing whitespace.
"""
function read_config_line(line)
    option_key, option_value = split_line(line, char="=")
    option_key = strip(option_key, ' ')
    option_value = strip(option_value, ' ')
    return option_key, option_value
end

"""
    filter_comment(line::String)

Removes comments from a line of text based on predefined comment symbols. The function scans for comment symbols, and if found, it removes the comment portion and returns the remaining part of the line.

# Arguments:
- `line::String`: The line of text from which comments need to be filtered out.

# Returns:
- A `String` that contains the line without comments. If the entire line is a comment or becomes empty after removing the comment, an empty string is returned.
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