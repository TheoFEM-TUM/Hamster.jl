"""
    parse_commandline(args::Vector{String}) -> Dict{String, String}

Parses command-line arguments from a vector of strings `args` and returns a dictionary of parsed arguments.

# Arguments
- `args`: A vector of command-line arguments passed as strings.

# Behavior
- Keyword arguments (`--option`): If an argument starts with `--`, it is treated as a key with an associated value in the following position. If a comma is found at the end of an argument, the following arguments are concatenated until no comma is found.
- Flags (`-o`): If an argument starts with a single `-`, it is treated as a flag and is set to `true` (empty "--" arguments are also treated as flags).
- Positional arguments: stored with key `pos_arg_i`.

# Returns
- `args_dict`: A dictionary containing parsed command-line arguments. Long options are stored as key-value pairs, flags are stored with a value of `true`.
"""
function parse_commandline(args)
    num_pos = 0
    args_dict = Dict{String, String}()
    for (k, arg) in enumerate(args)
        if arg == "-h" || arg == "--help"
            args_dict["help"] = "true"
        elseif occursin("--", arg)
            new_arg = (length(args) > k && args[k+1][1] ≠ '-') ? args[k+1] : "true"
            j = 0
            while k+j+1 < length(args) && args[k+1+j][end] == ','
                new_arg *= args[k+2+j]
                j += 1
            end

            args_dict[arg[3:end]] = new_arg
        elseif arg[1] == '-'
            args_dict[arg[2:end]] = "true"
        elseif k == 1 || (k > 1 ? !occursin("--", args[k-1]) : false) || args[k-1] == "--help"
            num_pos += 1
            args_dict["pos_arg_$num_pos"] = arg
        end
    end
    return args_dict
end