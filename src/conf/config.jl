"""
    struct Config

A configuration structure that holds options and blocks of tags in dictionaries.

# Fields
- `options::Dict{String, String}`: A dictionary holding configuration options.
- `blocks::Dict{String, Dict{String, String}}`: A dictionary holding configuration blocks.
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
    lowercase_key = lowercase(key)
    if haskey(conf.options, lowercase_key) && typekey == "none"
        return convert_value(conf.options[lowercase_key])
    elseif typekey ≠ "none" && haskey(conf.blocks, typekey)
        if haskey(conf.blocks[typekey], lowercase_key)
            return convert_value(conf.blocks[typekey][lowercase_key])
        else
            return "default"
        end
    else
        return "default"
    end
end

"""
    ConfigTag{T}

An input tag of the config file: its `name`, the `block` it belongs to (`"Options"`, `"Supercell"`, ..., `"atom"` for the
`begin atom <element>` blocks), the function `default(conf)` that gives its default value (of type `T`), a description,
the default in words for the documentation (`default_doc`), a `category` within the block, related tags (`see_also`)
and whether it is set per atom species (`per_atom`). All tags are collected in `CONFIG_TAGS`, from which the
documentation, the listing in `hamster.out`, `hamster --tag` and the checks of the config file are generated.
"""
struct ConfigTag{T,F}
    name::String
    block::String
    default::F           # getter conf -> T
    description::String
    default_doc::String
    category::String
    see_also::Vector{String}
    per_atom::Bool
end

ConfigTag{T}(name::String, default::F, desc::String; kwargs...) where {T,F} = ConfigTag{T}(name, "Options", default, desc; kwargs...)

ConfigTag{T}(name::String, block::String, default::F, desc::String; default_doc="", category="", see_also=String[],
             per_atom=false) where {T,F} =
    ConfigTag{T,F}(name, block, default, desc, default_doc, category, collect(String, see_also), per_atom)

tag_type(::ConfigTag{T}) where {T} = T

function get_tag(conf::Config, tag::ConfigTag{T})::T where T
    if tag.block == "Options"
        return get(conf, tag.name, tag.default(conf))
    else
        return get(conf, tag.name, tag.block, tag.default(conf))
    end
end

"""
    tag_value(conf, tag)

Value of `tag` in `conf` for the listing in hamster.out: the value set in the config file, else the default.
"""
function tag_value(conf::Config, tag::ConfigTag)
    v = tag.block == "Options" ? conf(tag.name) : conf(tag.name, tag.block)
    return v == "default" ? tag.default(conf) : v
end

const CONFIG_TAGS = ConfigTag[]

# The default of a tag in words, if it is a literal (number, string, vector); "" for computed defaults, which need a
# `default_doc`
function literal_default_doc(default)
    default isa Union{Number, AbstractString} && return string(default)
    default isa QuoteNode && return string(default.value)
    if default isa Expr
        default.head == :vect && return isempty(default.args) ? "(empty)" : join(string.(default.args), " ")
        default.head == :ref && length(default.args) == 1 && return "(empty)"        # e.g. String[]
        default.head == :call && default.args[1] == :- && length(default.args) == 2 && default.args[2] isa Number &&
            return "-" * string(default.args[2])
    end
    return ""
end

"""
    @configtag name T default "description" ["Block"] [default_doc="..."] [category="..."] [see_also=[...]] [getter=:custom] [per_atom=true] [value=f]

Define the input tag `name` of type `T` in `Block` (default: `"Options"`) with the default value `default`, which may
depend on other tags through `conf`, and register it in `CONFIG_TAGS`. A getter `get_<name>(conf)` (for the Options,
Supercell and Optimizer blocks) or `get_<block>_<name>(conf)` is generated, unless `getter=:custom` (the getter is then
written by hand, e.g. for tags with their own logic, or `per_atom=true` for the tags of the atom blocks).

- `default_doc`: the default in words for the documentation, required if `default` is not a literal (e.g.
  `"same as rcut"`).
- `category`: group within the block in the documentation (e.g. `"structure"`).
- `see_also`: names of related tags.
- `value`: function `conf -> value` for the listing in `hamster.out`, for custom getters that need further arguments
  or return another type.
"""
macro configtag(name, T, default, desc, args...)
    block = "Options"; kw = Dict{Symbol, Any}()
    for a in args
        if a isa String
            block = a
        elseif a isa Expr && a.head == :(=)
            kw[a.args[1]] = a.args[2]
        else
            error("@configtag $name: unexpected argument $a")
        end
    end
    custom = haskey(kw, :getter) && (kw[:getter] isa QuoteNode ? kw[:getter].value : kw[:getter]) == :custom
    per_atom = get(kw, :per_atom, false)
    default_doc = get(kw, :default_doc, literal_default_doc(default))
    category = get(kw, :category, "")
    see_also = get(kw, :see_also, :(String[]))
    fname = block ∈ ["Options", "Supercell", "Optimizer"] ? Symbol("get_", name) : Symbol("get_$(lowercase(block))_", name)
    tagname = string(name)
    default_fn = (custom || per_atom) ? :(conf -> $(fname)(conf)) : :(conf -> $default)
    per_atom && (default_fn = :(conf -> nothing))
    haskey(kw, :value) && (default_fn = kw[:value])

    docstring = """
        $tagname = $(isempty(default_doc) ? string(default) : default_doc)   ($block block, $T)

    $desc
    """

    tagvar = gensym("tag")      # unique per call, so that getters defined in the same scope do not share the variable
    getter = (custom || per_atom) ? nothing : quote
        """
            $(esc($docstring).args[1])
        """
        function $(fname)(conf::Config)::$T
            get_tag(conf, $tagvar)
        end
    end

    return quote
        local $tagvar = ConfigTag{$T}($tagname, $block, $default_fn, $desc; default_doc=$default_doc, category=$category,
                                      see_also=$see_also, per_atom=$per_atom)
        push!(CONFIG_TAGS, $tagvar)
        $getter
    end |> esc
end

"""
    Base.get(conf::Config, key, default::T) :: T where {T}
     Base.get(conf::Config, key, typekey, default::T) :: T where {T}

Retrieve a value associated with `key` (and `typekey`) from a configuration object `conf`. If the value
returned by `conf(key)` is `"default"`, the fallback value `default` is returned instead.

# Arguments
- `conf::Config`: A configuration object, typically implementing callable behavior for key lookups.
- `key`: The key to look up in the configuration. Its type depends on the `Config` implementation.
- `default::T`: A fallback value of type `T` to return if the configuration value for `key` is `"default"`.

# Returns
- `::T`: The value associated with `key` in `conf`, unless that value is `"default"`, in which case
  `default` is returned.
"""
function Base.get(conf::Config, key, default::T)::T where {T}
    if conf(key) == "default"
        return default
    else
        val = conf(key)
        if T <: AbstractVector && !(val isa AbstractVector)
            return [val]
        else
            return val
        end
    end
end

function Base.get(conf::Config, key, typekey, default::T)::T where {T}
    if conf(key, typekey) == "default"
        return default
    else
        val = conf(key, typekey)
        if T <: AbstractVector && !(val isa AbstractVector)
            return [val]
        else
            return val
        end
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
Base.haskey(conf::Config, key) = haskey(conf.options, lowercase(key)) || haskey(conf.blocks, key)

"""
    set_value!(conf::Config, key, value)

Sets the value of a given key in the `options` dictionary of a `Config` instance.

# Arguments
- `key`: The key for which the value needs to be set.
- `value`: The value to set for the specified key.
- `conf::Config`: The `Config` instance where the key-value pair should be set.
"""
set_value!(conf::Config, key, value) = conf.options[lowercase(key)] = string(value)

"""
    set_value!(conf::Config, key, typekey, value)

Sets the value of a given key in the specified block of the `blocks` dictionary of a `Config` instance. If the block does not exist, it creates a new block with the given key-value pair.
Keys are always stored in lowercase.

# Arguments
- `key`: The key to set in the specified block.
- `typekey`: The key for the block in which the key-value pair should be set.
- `value`: The value to set for the specified key.
- `conf::Config`: The `Config` instance where the key-value pair should be set.
"""
function set_value!(conf::Config, key, typekey, value)
    lowercase_key = lowercase(key)
    if haskey(conf.blocks, typekey)
        conf.blocks[typekey][lowercase_key] = string(value)
    else
        conf.blocks[typekey] = Dict{String, String}(lowercase_key=>string(value))
    end
end

"""
    convert_value(value::String)

Attempts to convert a string `value` into its most appropriate data type among `Int64`, `Float64`, and `Bool`. If the string represents a valid value for one of these types, it is parsed and returned as that type. If not, the original string is returned.

# Arguments:
- `value::String`: The string representation of a value to be converted.

# Returns:
- The converted value in its most appropriate type (either `Int64`, `Float64`, or `Bool`), or the original string if it cannot be converted.

# Behavior:
- The function splits the input `value` based on spaces into components using `split_line(value, char=' ')`.
- It then attempts to parse these components into `Int64`, `Float64`, and `Bool` types.
"""
function convert_value(value)
    n_semicolon = count(c -> c == ';', value)
    values = split_line(value, char=[' ', ';'])
    for type in [Int64, Float64, Bool]
        if check_parse(type, values)
            if n_semicolon > 0
                return transpose(reshape(parse_value(type, values), (:, n_semicolon+1)))
            else
                return parse_value(type, values)
            end
        end
    end
    if length(values) == 1 
        return values[1]
    else
        return values
    end
end

"""
    check_parse(type, line)

Check if the string or vector of strings `line` can be converte to `type`.
"""
function check_parse(type, value)
    valueparse = parse_value(type, value)
    return all(x->x≠nothing, valueparse)
end

function parse_value(type, value)
    valueparse = tryparse.(type, lowercase.(value))
    if all(x->x≠nothing, valueparse)
        if length(valueparse) == 1; return valueparse[1]; end
    end
    return valueparse
end