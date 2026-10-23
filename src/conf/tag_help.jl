# Functions on the registry of input tags (CONFIG_TAGS): lookup with suggestions for typos, the help of `hamster --tag`,
# the check of a config file for unknown and misplaced tags, and the generated documentation of all tags.

"""
Blocks of the config file in the order of the documentation, with a short description and the order of the categories
of their tags.
"""
const CONFIG_BLOCKS = [
    ("Options", "General settings of every run.",
     ["input/output", "structure", "basis", "model", "diagonalization", "device and parallelization", "post-processing"]),
    ("Supercell", "Supercells and trajectories (e.g. MD snapshots); switches a standard run to supercell mode.",
     ["structures"]),
    ("Optimizer", "Parameter fits; the presence of the block switches to an optimization run.",
     ["data", "parameters", "bands and weights", "loss and learning rate", "validation", "output"]),
    ("HyperOpt", "Hyperparameter optimization (together with an Optimizer block).", ["search"]),
    ("ML", "Machine-learned corrections (kernel model); the presence of the block adds the model.",
     ["model", "parameters", "descriptor", "data points", "precomputed similarities"]),
    ("SOC", "Spin-orbit coupling; the presence of the block (or soc = true) adds the model.", ["model"]),
    ("Ewald", "Electrostatic on-site corrections; the presence of the block (or ewald = true) adds the model.",
     ["model", "electrostatics"]),
    ("DOS", "DOS, PDOS, COHP and JDOS post-processing of a standard run (switched on by the dos tag in Options).",
     ["quantities", "method", "energy grid and broadening", "electrons", "JDOS", "output"]),
    ("atom", "Settings per atom species, in blocks `begin atom <element>` (e.g. `begin atom Pb`).",
     ["basis", "electrostatics"]),
]

block_order(block) = something(findfirst(b -> b[1] == block, CONFIG_BLOCKS), length(CONFIG_BLOCKS) + 1)
function category_order(tag)
    i = block_order(tag.block)
    i > length(CONFIG_BLOCKS) && return 0
    return something(findfirst(==(tag.category), CONFIG_BLOCKS[i][3]), length(CONFIG_BLOCKS[i][3]) + 1)
end

"""
    config_tags(; block=nothing) -> Vector{ConfigTag}

All registered input tags (of `block`), in the order of the documentation (block, category, name).
"""
function config_tags(; block=nothing)
    tags = filter(tag -> isnothing(block) || tag.block == block, CONFIG_TAGS)
    return sort(tags, by=tag -> (block_order(tag.block), category_order(tag), lowercase(tag.name)))
end

"""
    find_tags(name; block=nothing) -> Vector{ConfigTag}

The tags with the name `name` (not case sensitive), in all blocks or only in `block`.
"""
find_tags(name; block=nothing) = filter(tag -> lowercase(tag.name) == lowercase(name) && (isnothing(block) || tag.block == block),
                                        config_tags())

# Optimistic string distance (Damerau-Levenshtein with adjacent transpositions) for suggestions
function string_distance(a::AbstractString, b::AbstractString)
    a, b = collect(lowercase(a)), collect(lowercase(b))
    m, n = length(a), length(b)
    d = [i == 0 ? j : j == 0 ? i : 0 for i in 0:m, j in 0:n]
    for i in 1:m, j in 1:n
        c = a[i] == b[j] ? 0 : 1
        d[i+1, j+1] = min(d[i, j+1] + 1, d[i+1, j] + 1, d[i, j] + c)
        i > 1 && j > 1 && a[i] == b[j-1] && a[i-1] == b[j] && (d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + 1))
    end
    return d[m+1, n+1]
end

"""
    suggest_tags(name; block=nothing, n=3) -> Vector{ConfigTag}

Up to `n` tags whose names are close to `name` (typos), preferring tags of `block`.
"""
function suggest_tags(name; block=nothing, n=3)
    tol = max(2, length(name) ÷ 3)
    scored = [(string_distance(name, tag.name) + (isnothing(block) || tag.block == block ? 0 : 0.5), tag) for tag in config_tags()]
    filter!(s -> s[1] ≤ tol, scored)
    sort!(scored, by=first)
    seen = Set{String}(); out = ConfigTag[]
    for (_, tag) in scored                                 # each name once, from the best-matching block
        lowercase(tag.name) ∈ seen && continue
        push!(seen, lowercase(tag.name)); push!(out, tag)
        length(out) == n && break
    end
    return out
end

tag_location(tag) = tag.block == "atom" ? "atom block" : "$(tag.block) block"
default_text(tag) = isempty(tag.default_doc) ? "(see description)" : tag.default_doc

"""
    tag_help([io], name)

Print the description of the input tag `name` (all blocks in which it exists), or suggestions if there is no such tag.
"""
tag_help(name) = tag_help(stdout, name)
function tag_help(io::IO, name)
    tags = find_tags(name)
    if isempty(tags)
        suggestions = suggest_tags(name)
        println(io, "No input tag \"$name\".", isempty(suggestions) ? "" :
                " Did you mean: " * join(["$(t.name) ($(tag_location(t)))" for t in suggestions], ", ") * "?")
        return nothing
    end
    for (k, tag) in enumerate(tags)
        k > 1 && println(io)
        println(io, "$(tag.name)  ($(tag_location(tag)), $(tag_type(tag))), default: $(default_text(tag))")
        for line in wrap_text(tag.description, 100); println(io, "  ", line); end
        isempty(tag.category) || println(io, "  Category: $(tag.category).")
        isempty(tag.see_also) || println(io, "  See also: $(join(tag.see_also, ", ")).")
    end
    return nothing
end

"""
    list_tags([io], filter="")

Print all input tags of a block (`filter` = block name, e.g. `"ML"`), or all tags whose name or description contains
`filter` (empty: all tags), one line each.
"""
list_tags(filter_term="") = list_tags(stdout, filter_term)
function list_tags(io::IO, filter_term="")
    blocks = [b[1] for b in CONFIG_BLOCKS]
    term = lowercase(filter_term)
    block = findfirst(b -> lowercase(b) == term, blocks)
    tags = !isnothing(block) ? config_tags(block=blocks[block]) :
           filter(t -> isempty(term) || occursin(term, lowercase(t.name)) || occursin(term, lowercase(t.description)), config_tags())
    isempty(tags) && (println(io, "No input tags match \"$filter_term\"."); return nothing)
    width = maximum(length(t.name) for t in tags) + 2
    current = ""
    for tag in tags
        if tag.block != current
            current = tag.block; println(io, "\n[$current]")
        end
        println(io, "  ", rpad(tag.name, width), first(wrap_text(tag.description, 90)))
    end
    return nothing
end

"""
    tag_cli(args)

Entry point of `hamster --tag NAME ...` (description of each tag) and `hamster --tags [BLOCK|TERM]` (list of tags).
"""
function tag_cli(args)
    command = isempty(args) ? "--tags" : args[1]; rest = args[2:end]
    if command == "--tag"
        isempty(rest) && (println("Usage: hamster --tag NAME (e.g. hamster --tag rcut)"); return nothing)
        for (k, name) in enumerate(rest)
            k > 1 && println()
            tag_help(name)
        end
    else
        list_tags(join(rest, " "))
    end
    return nothing
end

# Keys of the Optimizer block with an index (band or k-point weights such as wE_3, wE_10-20, wk_1)
is_indexed_weight(key, block) = block == "Optimizer" && occursin(r"^w[ek]_[0-9]+(-[0-9]+)?$", lowercase(key))
is_atom_block(name) = haskey(elements, Symbol(name))

"""
    check_config_tags(conf) -> Vector{String}

Messages for the tags of `conf` that Hamster does not know or that are placed in a block where they have no effect,
with suggestions (e.g. `"xdatcar" belongs in the Supercell block`). Unknown blocks are reported as well.
"""
function check_config_tags(conf::Config)
    messages = String[]
    known_blocks = [b[1] for b in CONFIG_BLOCKS]
    function check(key, block, shown_block)
        isempty(find_tags(key, block=block)) || return
        is_indexed_weight(key, block) && return
        elsewhere = find_tags(key)
        if !isempty(elsewhere)
            push!(messages, "Input tag \"$key\" has no effect in the $shown_block block; it belongs in the " *
                            join(unique(tag_location.(elsewhere)), " or ") * ".")
        else
            suggestions = suggest_tags(key, block=block)
            push!(messages, "Unknown input tag \"$key\" in the $shown_block block" * (isempty(suggestions) ? "." :
                            "; did you mean " * join(["$(t.name) ($(tag_location(t)))" for t in suggestions], ", ") * "?"))
        end
    end
    for key in keys(conf.options)
        check(key, "Options", "Options")
    end
    for (name, block) in conf.blocks
        if is_atom_block(name)
            for key in keys(block); check(key, "atom", "atom $name"); end
        elseif name ∈ known_blocks
            for key in keys(block); check(key, name, name); end
        else
            push!(messages, "Unknown block \"$name\"; known blocks: $(join(known_blocks[1:end-1], ", ")) and atom <element>.")
        end
    end
    return messages
end

# ==============================================================================================================
# Documentation of all tags (generated by docs/make.jl)
# ==============================================================================================================

anchor(tag) = "tag-$(lowercase(tag.block))-$(lowercase(tag.name))"
block_file(block) = "tags_$(lowercase(block)).md"
md_escape(s) = replace(s, "|" => "\\|")

const BLOCK_IDS = Dict("Options" => "options-block", "Optimizer" => "optim-tags", "HyperOpt" => "hyperopt-tags",
                       "DOS" => "dos-tags")

"""
    write_tag_docs(dir)

Write the documentation of all input tags to `dir`: one page per block (`tags_<block>.md`, tags grouped by category,
with an anchor per tag) and an alphabetical index (`tags_index.md`). Returns the file names in the order of the pages.
"""
function write_tag_docs(dir)
    mkpath(dir)
    files = String[]
    for (block, desc, categories) in CONFIG_BLOCKS
        tags = config_tags(block=block)
        isempty(tags) && continue
        file = block_file(block); push!(files, file)
        open(joinpath(dir, file), "w") do io
            title = block == "atom" ? "atom blocks" : "$block block"
            println(io, "# [$title](@id $(get(BLOCK_IDS, block, "block-$(lowercase(block))")))\n")
            println(io, "*This page is generated from the tag definitions in `src/conf`.*\n")
            println(io, desc, "\n")
            println(io, "```bash")
            println(io, block == "Options" ? "begin Options" : block == "atom" ? "begin atom Pb" : "begin $block")
            println(io, "    $(tags[1].name) = ...\nend\n```\n")
            for category in vcat(categories, [""])
                ctags = filter(t -> t.category == category || (category == "" && t.category ∉ categories), tags)
                isempty(ctags) && continue
                println(io, "## ", isempty(category) ? "Other" : uppercasefirst(category), "\n")
                for tag in ctags
                    println(io, "### [`$(tag.name)`](@id $(anchor(tag)))\n")
                    println(io, "**Type:** `$(tag_type(tag))` | **Default:** $(default_text(tag))\n")
                    println(io, tag.description, "\n")
                    if !isempty(tag.see_also)
                        links = map(tag.see_also) do name
                            target = find_tags(name, block=tag.block); target = isempty(target) ? find_tags(name) : target
                            isempty(target) ? "`$name`" : "[`$name`](@ref $(anchor(target[1])))"
                        end
                        println(io, "See also: ", join(links, ", "), "\n")
                    end
                end
            end
        end
    end
    open(joinpath(dir, "tags_index.md"), "w") do io
        println(io, "# [All input tags](@id block-labels)\n")
        println(io, "*This page is generated from the tag definitions in `src/conf`.* The description of a tag is also ",
                    "shown by `hamster --tag <name>`, and `hamster --tags <block or term>` lists tags.\n")
        println(io, "| Tag | Block | Type | Default | Description |")
        println(io, "|---|---|---|---|---|")
        for tag in sort(config_tags(), by=t -> (lowercase(t.name), block_order(t.block)))
            println(io, "| [`$(tag.name)`](@ref $(anchor(tag))) | $(tag.block) | `$(tag_type(tag))` | ",
                        md_escape(default_text(tag)), " | ", md_escape(first(split(tag.description, ". "))), " |")
        end
    end
    return vcat(["tags_index.md"], files)
end
