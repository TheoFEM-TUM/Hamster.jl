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
Config(;filename="hconf") = read_config(filename)

"""
    read_config()

Read the tbconfig file if given, else return default config.
"""
function read_config(file::String)
    conf = get_empty_config()
    if isfile(file)
        lines = open_and_read(file)
        blocks = split_blocks(lines, source=file)
        for block in blocks
            read_config_block!(block, conf, source=file)
        end
        # Unknown and misplaced tags (warned once, on the first MPI rank)
        if !MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0
            foreach(message -> @warn(message * " See `hamster --tag <name>` or the list of input tags in the documentation."),
                    check_config_tags(conf))
        end
        assert_valid_config(conf, source=file)
    else
        @info "File $file not found. Using default values."
    end
    return conf
end

"""
    split_blocks(lines; source="config")

Splits a list of lines into blocks of text based on "begin" and "end" markers, i.e., lines whose first word (without
comments) is `begin` or `end`. Each block is encapsulated in a `ConfigBlock` struct. A block without a name or
without `end`, a `begin` inside a block and an `end` outside a block raise an error with the line number in `source`.

# Arguments
- `lines`: A list of strings, where each string is a line from the input text.

# Returns
- A list of `ConfigBlock` instances, each containing the header and content of a block found in the input lines.
"""
function split_blocks(lines; source="config")
    blocks = ConfigBlock[]
    begin_line = 0
    for (k, line) in enumerate(lines)
        words = split_line(filter_comment(line))
        first_word = isempty(words) ? "" : words[1]
        if first_word == "begin"
            begin_line == 0 || error("$source, line $k: \"begin\" inside the block of line $begin_line (missing \"end\"?).")
            length(words) > 1 || error("$source, line $k: \"begin\" without a block name.")
            begin_line = k
        elseif first_word == "end"
            begin_line > 0 || error("$source, line $k: \"end\" without \"begin\".")
            header = string.([el for el in split_line(filter_comment(lines[begin_line])) if el ≠ "begin"])
            push!(blocks, ConfigBlock(header, lines[begin_line+1:k-1], begin_line))
            begin_line = 0
        end
    end
    begin_line == 0 || error("$source, line $begin_line: block without \"end\".")
    return blocks
end

"""
    ConfigBlock

A structure representing a block of configuration settings.

The `ConfigBlock` struct is used to encapsulate a segment of a configuration file, typically consisting of a header and associated content lines. The header usually indicates the section or category of the configuration, while the content contains key-value pairs or other relevant settings.

# Fields:
- `header::Vector{String}`: A vector of strings representing the header of the configuration block. This typically identifies the section or category the block belongs to.
- `content::Vector{String}`: A vector of strings representing the content of the configuration block. Each string usually corresponds to a line in the configuration file, often containing key-value pairs or other settings.
- `line::Int64`: The line number of the `begin` line of the block (for error messages).
"""
struct ConfigBlock
    header :: Vector{String}
    content :: Vector{String}
    line :: Int64
end

ConfigBlock(header, content) = ConfigBlock(header, content, 0)

"""
    read_config_block!(block::ConfigBlock, conf::Config; source="config")

Processes a configuration block and updates the configuration object with the parsed key-value pairs.

This function iterates over the lines in a `ConfigBlock`, parses each line to extract configuration options and their values, and updates the provided `Config` object. The function handles lines containing the `=` character, while ignoring comments and empty lines. The updates are made based on whether the block's header ends with `"Options"` or some other string.

# Arguments:
- `block::ConfigBlock`: An object representing a block of configuration lines, typically containing a header and a list of content lines.
- `conf::Config`: The configuration object that will be updated with the parsed key-value pairs from the block.

Lines without `=`, with more than one `=` or without a tag name or value raise an error with the line number in
`source`.
"""
function read_config_block!(block::ConfigBlock, conf::Config; source="config")
    for (k, line) in enumerate(block.content)
        text = filter_comment(line)
        isempty(strip(text)) && continue
        where = "$source, line $(block.line + k): \"$(strip(text))\""
        count(==('='), text) == 1 || error("$where is not of the form tag = value.")
        option_key, option_value = read_config_line(text)
        isempty(option_key) && error("$where has no tag name.")
        isempty(option_value) && error("$where has no value.")
        option_key = renamed_tag(option_key)
        if block.header[end] == "Options"
            set_value!(conf, option_key, option_value)
        else
            set_value!(conf, option_key, block.header[end], option_value)
        end
    end
end

"""
    RENAMED_TAGS

Input tags that were renamed (old name => new name). Config files with an old name keep working: the value is used for
the new tag, with a warning.
"""
const RENAMED_TAGS = Dict("diag_method" => "eigsolver")

"""
    renamed_tag(key)

The current name of the input tag `key` (see `RENAMED_TAGS`); warns (on the first MPI rank) if `key` is an old name.
"""
function renamed_tag(key)
    new_key = get(RENAMED_TAGS, lowercase(key), nothing)
    isnothing(new_key) && return key
    if !MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @warn "Input tag \"$key\" was renamed to \"$new_key\"; please update the config file."
    end
    return new_key
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
    option_key, option_value = split(line, '=', limit=2)
    return strip(option_key), strip(option_value)
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
    n = findfirst(c -> c ∈ COMMENT_SYMBOLS, line)
    return isnothing(n) ? line : strip(line[1:prevind(line, n)], ' ')
end