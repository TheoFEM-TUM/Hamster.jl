"""
    NeighborList

Directed periodic neighbor list: for every pair `p`, atom `j[p]` in the image `n[p]` (lattice vector `L n[p]`) is within
the cutoff of atom `i[p]`, with the bond vector `r[:, j] + L n - r[:, i]` of the (unwrapped) positions `r`. Every
unordered pair appears twice, as (i, j, n) and (j, i, -n). The list is valid as long as no atom has moved by more than
half of the skin since `positions`.

# Fields
- `i`, `j::Vector{Int32}`: atoms of the pairs, sorted by `i` (`first[a]:first[a+1]-1` are the pairs of atom `a`).
- `n::Vector{SVector{3, Int32}}`: lattice translations of `j`.
- `first::Vector{Int64}`: start of the pairs of each atom.
- `rc::Float64`: cutoff including the skin.
- `positions::Matrix{Float64}`: positions at which the list was built.
"""
struct NeighborList
    i :: Vector{Int32}
    j :: Vector{Int32}
    n :: Vector{SVector{3, Int32}}
    first :: Vector{Int64}
    rc :: Float64
    skin :: Float64
    positions :: Matrix{Float64}
end

Base.length(nl::NeighborList) = length(nl.i)

"""
    NeighborList(positions, lattice, rc; skin=0.0, rmin=0.0, include=(i, j) -> true)

Build the neighbor list of all pairs with `rmin < |r_j + L n - r_i| < rc + skin` (atoms in the 3 × N matrix
`positions`, Cartesian, lattice vectors in the columns of `lattice`), with cell lists (linear scaling) for any cell
shape and size, including cells smaller than the cutoff (several images of the same atom). `include(i, j)` filters the
pairs (e.g. atoms with orbitals).
"""
function NeighborList(positions, lattice, rc; skin=0.0, rmin=0.0, include=(i, j) -> true)
    N = size(positions, 2)
    L = SMatrix{3, 3, Float64}(lattice)
    Linv = inv(L)
    rlist = rc + skin
    # Number of cells per lattice direction with a width (perpendicular to the other two vectors) of at least rlist,
    # and how many cells have to be searched in each direction
    V = abs(det(L))
    widths = SVector{3}(V / norm(cross(L[:, 2], L[:, 3])), V / norm(cross(L[:, 3], L[:, 1])), V / norm(cross(L[:, 1], L[:, 2])))
    ncell = SVector{3, Int}(max(1, min(floor(Int, w / rlist), 64)) for w in widths)
    reach = SVector{3, Int}(ceil(Int, rlist / (widths[d] / ncell[d])) for d in 1:3)
    # Wrapped fractional coordinates and the lattice vectors that were removed by wrapping
    s = [Linv * SVector{3, Float64}(positions[1, a], positions[2, a], positions[3, a]) for a in 1:N]
    wraps = [SVector{3, Int}(floor.(Int, sa)) for sa in s]
    s = [s[a] - wraps[a] for a in 1:N]
    cell_of(sa) = SVector{3, Int}(min(ncell[d] - 1, floor(Int, sa[d] * ncell[d])) for d in 1:3)
    cells = Dict{SVector{3, Int}, Vector{Int}}()
    for a in 1:N
        push!(get!(cells, cell_of(s[a]), Int[]), a)
    end
    offsets = vec([SVector{3, Int}(x, y, z) for x in -reach[1]:reach[1], y in -reach[2]:reach[2], z in -reach[3]:reach[3]])
    rlist2 = rlist^2; rmin2 = rmin^2
    parts = tmap(chunks(1:N, n=4 * Threads.nthreads())) do atoms
        is = Int32[]; js = Int32[]; ns = SVector{3, Int32}[]
        for a in atoms
            ca = cell_of(s[a])
            ra = L * s[a]
            for off in offsets
                c = ca + off
                cw = SVector{3, Int}(mod(c[d], ncell[d]) for d in 1:3)
                shift = SVector{3, Int}(fld(c[d], ncell[d]) for d in 1:3)
                haskey(cells, cw) || continue
                for b in cells[cw]
                    (a == b && iszero(shift)) && continue
                    include(a, b) || continue
                    d = L * (s[b] + shift) - ra
                    d2 = dot(d, d)
                    if rmin2 < d2 < rlist2
                        push!(is, a); push!(js, b)
                        # Translation for the unwrapped positions: r_b - r_a + L n = (wrapped difference) + L shift
                        push!(ns, SVector{3, Int32}(shift - wraps[b] + wraps[a]))
                    end
                end
            end
        end
        (is, js, ns)
    end
    i = reduce(vcat, getindex.(parts, 1)); j = reduce(vcat, getindex.(parts, 2)); n = reduce(vcat, getindex.(parts, 3))
    first = zeros(Int64, N + 1); first[1] = 1
    for a in i
        first[a+1] += 1
    end
    cumsum!(first, first)
    return NeighborList(i, j, n, first, rc, skin, copy(positions))
end

"""
    bond_vector(positions, lattice, i, j, n)

Bond vector `r_j + L n - r_i` of a pair.
"""
@inline function bond_vector(positions, L::SMatrix{3, 3}, i, j, n)
    return SVector{3, Float64}(positions[1, j] - positions[1, i], positions[2, j] - positions[2, i],
                               positions[3, j] - positions[3, i]) + L * n
end

"""
    needs_rebuild(nl, positions)

Whether an atom has moved by more than half of the skin since the list was built.
"""
function needs_rebuild(nl::NeighborList, positions)
    lim2 = (nl.skin / 2)^2
    return tmapreduce(|, axes(positions, 2)) do a
        Δ = SVector{3}(positions[1, a] - nl.positions[1, a], positions[2, a] - nl.positions[2, a], positions[3, a] - nl.positions[3, a])
        dot(Δ, Δ) > lim2
    end
end

"""
    update_neighbors(nl, positions, lattice; kwargs...)

The neighbor list `nl` if it is still valid for `positions`, otherwise a new one with the same cutoff and skin.
"""
function update_neighbors(nl::NeighborList, positions, lattice; kwargs...)
    needs_rebuild(nl, positions) || return nl
    return NeighborList(positions, lattice, nl.rc; skin=nl.skin, kwargs...)
end

"""
    reverse_pairs(nl)

Index of the reversed pair (j, i, -n) of every pair of the neighbor list.
"""
function reverse_pairs(nl::NeighborList)
    index = Dict{Tuple{Int32, Int32, SVector{3, Int32}}, Int32}()
    for p in eachindex(nl.i)
        index[(nl.i[p], nl.j[p], nl.n[p])] = p
    end
    return Int32[index[(nl.j[p], nl.i[p], -nl.n[p])] for p in eachindex(nl.i)]
end
