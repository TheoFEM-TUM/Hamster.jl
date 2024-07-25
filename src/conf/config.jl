"""
    struct Config{S1, S2, D}

A configuration structure that holds options and blocks of tags in dictionaries.

# Fields
- `options::Dict{S1, S2}`: A dictionary holding configuration options.
- `blocks::Dict{S1, D}`: A dictionary holding configuration blocks.
"""
struct Config
    options :: Dict{String, String}
    blocks :: Dict{String, Dict{String, String}}
end

"""
    get_empty_config()

Returns an empty `Config` instance.

# Returns
- `Config`: An empty `Config` instance with empty `options` and `blocks` dictionaries.
"""
get_empty_config()::Config = Config(Dict{String, String}(), Dict{String, Dict{String, String}}())

"""
    (conf::Config)(key, typekey="none")

Retrieve a value associated with a given key from `conf`. If a `typekey` is provided and exists in the `blocks` dictionary, it retrieves the value from the corresponding block. 
If no `typekey` is provided or it is set to `"none"`, it retrieves the value from the `options` dictionary.

# Arguments
- `conf::Config`: The `Config` instance to query.
- `key`: The key to look up in the `Config` instance.
- `typekey` (optional): The key for the block in the `blocks` dictionary to look up the value. Defaults to `"none"`.

# Returns
- The value associated with the `key` in the `options` dictionary if `typekey` is `"none"` and the key exists.
- The value associated with the `key` in the specified block of the `blocks` dictionary if `typekey` is provided and exists, and the key exists in that block.
- The string "default" if the key does not exist in the relevant dictionary or block.
"""
function (conf::Config)(key::String, typekey="none")
    if haskey(conf.options, key) && typekey == "none"
        return conf.options[key]
    elseif typekey ≠ "none" && haskey(conf, typekey)
        if haskey(conf.blocks[typekey], key)
            return conf.blocks[typekey][key]
        else
            return "default"
        end
    else 
        return "default"
    end
end

"""
    (conf::Config)(keys::Vector{String}, typekey="none")

Makes the `Config` instance callable with a vector of keys to retrieve corresponding values. 
If a `typekey` is provided and exists in the `blocks` dictionary, it retrieves the values from the corresponding block. 
If no `typekey` is provided or it is set to `"none"`, it retrieves the values from the `options` dictionary.

# Arguments
- `conf::Config`: The `Config` instance to query.
- `keys::Vector{String}`: A vector of keys to look up in the `Config` instance.
- `typekey` (optional): The key for the block in the `blocks` dictionary to look up the values. Defaults to `"none"`.

# Returns
- A vector of values corresponding to the provided keys. 
  - If `typekey` is `"none"` and the key exists in the `options` dictionary, the value from the `options` dictionary is returned.
  - If `typekey` is provided and exists in the `blocks` dictionary, and the key exists in that block, the value from the block is returned.
  - If the key does not exist in the relevant dictionary or block, the string "default" is returned for that key.
"""
(conf::Config)(keys::Vector{String}, typekey="none") = [conf(key, typekey) for key in keys]

"""
    haskey(conf::Config, key)

Checks if a given key exists in either the `options` or `blocks` dictionary of a `Config` instance.

# Arguments
- `conf::Config`: The `Config` instance to check.
- `key`: The key to check for existence.

# Returns
- `Bool`: `true` if the key exists in either the `options` or `blocks` dictionary, `false` otherwise.
"""
Base.haskey(conf::Config, key) = haskey(conf.options, key) || haskey(conf.blocks, key)

"""
    set_value!(key, value, conf::Config)

Sets the value of a given key in the `options` dictionary of a `Config` instance.

# Arguments
- `key`: The key for which the value needs to be set.
- `value`: The value to set for the specified key.
- `conf::Config`: The `Config` instance where the key-value pair should be set.
"""
set_value!(key, value, conf::Config) = conf.options[key] = value

"""
    set_value!(key, typekey, value, conf::Config)

Sets the value of a given key in the specified block of the `blocks` dictionary of a `Config` instance. If the block does not exist, it creates a new block with the given key-value pair.

# Arguments
- `key`: The key to set in the specified block.
- `typekey`: The key for the block in which the key-value pair should be set.
- `value`: The value to set for the specified key.
- `conf::Config`: The `Config` instance where the key-value pair should be set.
"""
function set_value!(key, typekey, value, conf::Config)
    if haskey(conf.blocks, typekey)
        conf.blocks[typekey][key] = value
    else
        conf.blocks[typekey] = Dict{String, String}(key=>value)
    end
end