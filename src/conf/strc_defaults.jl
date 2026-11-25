# ====================
# Structure defaults
# ====================
@configtag rcut Float64 7.0 "Distance beyond which all interactions are neglected; `rcut_<A>_<B>` (e.g. `rcut_Pb_Pb = 0`) sets a smaller cutoff for a pair of elements." category="structure" see_also=["rcut_tol", "grid_size"]
@configtag rcut_tol Float64 1.0 "offset (zero point) of the cutoff function w.r.t. `rcut`." category="structure" see_also=["rcut", "rcut_pair_selection"]
@configtag rcut_pair_selection String "reference" "positions the pair cutoff `rcut` is applied to: reference (reference structure, e.g. the first frame of each system) or frame (current positions; H depends only on the current geometry)." category="structure" values=["reference", "frame"] see_also=["rcut", "rcut_tol", "grid_size"]
@configtag Rmax Int64 1 "maximum magnitude of translation vectors used for periodic boundaty conditions (default: determined by rcut)." category="structure"
@configtag poscar String "POSCAR" "path to POSCAR file to define base system." category="structure"
@configtag grid_size Float64 get_rcut(conf) "size of the grid cells of the neighbor search; must be at least rcut." category="structure" default_doc="same as rcut"

"""
    pair_rcut_elements(key) -> Union{Tuple{String, String}, Nothing}

The two elements (sorted) of a pair cutoff tag `rcut_<A>_<B>` of the Options block (e.g. `rcut_Pb_Pb`, `rcut_Br_Pb`),
or `nothing` for other keys.
"""
function pair_rcut_elements(key)
    m = match(r"^rcut_([a-z]{1,2})_([a-z]{1,2})$", lowercase(key))
    isnothing(m) && return nothing
    a, b = uppercasefirst(m[1]), uppercasefirst(m[2])
    (haskey(elements, Symbol(a)) && haskey(elements, Symbol(b))) || return nothing
    return a ≤ b ? (a, b) : (b, a)
end

"""
    get_pair_rcuts(conf) -> Dict{Tuple{String, String}, Float64}

The cutoffs of the pairs of elements set with `rcut_<A>_<B>` in the Options block (sorted element pairs; the order of
`A` and `B` does not matter). A pair cutoff must lie between 0 and `rcut`; 0 removes all interactions between
different ions of the pair (on-site terms stay). Pairs without a tag use `rcut`.
"""
function get_pair_rcuts(conf::Config)
    rcut = get_rcut(conf)
    pairs = Dict{Tuple{String, String}, Float64}()
    for key in sort(collect(keys(conf.options)))
        pair = pair_rcut_elements(key)
        isnothing(pair) && continue
        value = conf(key)
        value isa Real || error("rcut_$(pair[1])_$(pair[2]) = $value is not a number.")
        0 ≤ value ≤ rcut || error("rcut_$(pair[1])_$(pair[2]) = $value must be between 0 and rcut = $rcut.")
        if haskey(pairs, pair) && pairs[pair] ≠ value
            error("The cutoff of the pair $(pair[1])-$(pair[2]) is set twice with different values ($(pairs[pair]) and $value).")
        end
        pairs[pair] = Float64(value)
    end
    return pairs
end

"""
    get_rcut_matrix(conf; rcut=get_rcut(conf)) -> Matrix{Float64}

Cutoff of every pair of atomic numbers (`M[Z1, Z2]`): the pair cutoffs `rcut_<A>_<B>` of the Options block
([`get_pair_rcuts`](@ref)), at most `rcut`, and `rcut` for all other pairs.
"""
function get_rcut_matrix(conf::Config=get_empty_config(); rcut=get_rcut(conf))
    M = fill(Float64(rcut), 118, 118)
    for ((a, b), value) in get_pair_rcuts(conf)
        za, zb = element_to_number(a), element_to_number(b)
        M[za, zb] = M[zb, za] = min(value, rcut)
    end
    return M
end
