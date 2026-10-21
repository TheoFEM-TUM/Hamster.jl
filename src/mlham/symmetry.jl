# ===========================================================================================================
# Point groups for symmetric ML corrections (enforce_symmetry_group)
# ===========================================================================================================
#
# A matrix element is linear in both orbitals, the kernel is not. The corrections are made exactly symmetric under a
# point group G by projecting the kernel onto functions that are linear in the orbitals, using a finite set of orbitals
# per shell that G maps onto itself: the orbit P of the basis orbitals under G (and the sign -1). With the frame
# operator S = Σ_{u ∈ P} u uᵀ (which commutes with the representation of G),
#
#     ΔH(c_i, c_j) = Σ_{u ∈ P_i} Σ_{v ∈ P_j} (S_i⁻¹ u)·c_i (S_j⁻¹ v)·c_j  K(x(u, v)),
#
# where c_i is the coefficient vector of orbital i in its shell and x(u, v) the descriptor of the orbitals u and v. If
# the descriptor is invariant under a joint rotation of the structure and the orbitals, this transforms like the
# Hamiltonian under every operation of G (for any structure, so it is also continuous in the geometry). For p and t2g
# orbitals under O_h the orbits are the signed basis orbitals (the sign flips of the orbitals); eg orbitals and the
# in-plane p orbitals under D_6h have hexagonal orbits, which also covers their mixing under the threefold/sixfold axes.

"""
    point_group(name) -> Vector{SMatrix{3, 3, Float64}}

Operations of the point group `name` (`"O_h"`: 48, `"D_6h"`: 24 with the sixfold axis along z), as orthogonal
matrices in Cartesian coordinates.
"""
function point_group(name)
    c, s = 0.5, √3 / 2
    generators = if name == "O_h"
        [SMatrix{3, 3}(0.0, 1, 0, -1, 0, 0, 0, 0, 1),            # C4 about z
         SMatrix{3, 3}(0.0, 1, 0, 0, 0, 1, 1, 0, 0),             # C3 about [111]: x → y → z → x
         SMatrix{3, 3}(-1.0, 0, 0, 0, -1, 0, 0, 0, -1)]          # inversion
    elseif name == "D_6h"
        [SMatrix{3, 3}(c, s, 0, -s, c, 0, 0, 0, 1),               # C6 about z
         SMatrix{3, 3}(1.0, 0, 0, 0, -1, 0, 0, 0, -1),           # C2 about x
         SMatrix{3, 3}(-1.0, 0, 0, 0, -1, 0, 0, 0, -1)]          # inversion
    else
        error("Unknown symmetry group $name (enforce_symmetry_group = O_h, D_6h or none).")
    end
    group = [SMatrix{3, 3}(1.0I)]
    known(g) = any(h -> isapprox(h, g, atol=1e-10), group)
    frontier = copy(group)
    while !isempty(frontier)
        new = SMatrix{3, 3, Float64, 9}[]
        for g in frontier, a in generators
            h = a * g
            known(h) || (push!(group, h); push!(new, h))
        end
        frontier = new
    end
    return group
end

# Fixed points on the unit sphere (Fibonacci lattice) to fit the representation matrices
const SPHERE_POINTS = [begin
    z = 1 - (2k - 1) / 100; r = √(1 - z^2); φ = k * π * (3 - √5)
    SVector(r * cos(φ), r * sin(φ), z)
end for k in 1:100]

"""
    shell_representation(types, g) -> Matrix{Float64}

Representation matrix D of the operation `g` on the orbitals `types` of a shell (angular functions): the rotated
orbital f_i(g⁻¹ r) = Σ_k D[k, i] f_k(r) (fitted on points of the sphere; exact up to rounding).
"""
function shell_representation(types, g)
    F = [t(p...) for p in SPHERE_POINTS, t in types]
    G = [t((g' * p)...) for p in SPHERE_POINTS, t in types]
    D = F \ G
    norm(F * D - G) < 1e-8 * norm(G) || error("The orbitals $(types) are not closed under the symmetry operation $g.")
    return D
end

"""
    ShellOrbit

Orbit of the orbitals of a shell (pure orbitals of one l, in the order of `types`) under a point group and the sign:
the vectors `U` (coefficients in the shell), the inverse frame operator `Sinv` = (Σ u uᵀ)⁻¹ and its square root
`Shalf` = (Σ u uᵀ)^(-1/2).
"""
struct ShellOrbit
    types :: Vector{Angular}
    U :: Vector{Vector{Float64}}
    Sinv :: Matrix{Float64}
    Shalf :: Matrix{Float64}
end

function ShellOrbit(types, group)
    n = length(types)
    U = Vector{Float64}[]
    add!(u) = any(v -> norm(v - u) < 1e-8, U) || push!(U, u)
    Ds = [shell_representation(types, g) for g in group]
    # Orbits of the basis orbitals that are not yet spanned (e.g. under O_h: dxz covers t2g, dz2 covers eg), which
    # gives the smallest set of this kind
    for k in 1:n
        e = zeros(n); e[k] = 1
        isempty(U) || norm(e - reduce(hcat, U) * (reduce(hcat, U) \ e)) > 1e-8 || continue
        for D in Ds, sgn in (1, -1)
            add!(round.(sgn * D * e, digits=12))
        end
    end
    S = sum(u * u' for u in U)
    return ShellOrbit(collect(types), U, inv(S), inv(sqrt(Symmetric(S))))
end

"""
    shell_orbits(basis, group_name) -> Dict{Int64, ShellOrbit}

Orbits of the shells with l ≥ 1 that occur in the basis (pure orbitals), by l.
"""
function shell_orbits(basis, group_name)
    group = point_group(group_name)
    ls = unique(orb.type.l for orbs in basis.orbitals for orb in orbs if orb.type.l ≥ 1)
    return Dict(l => ShellOrbit(shell_types(l), group) for l in ls)
end

"""
    shell_types(l)

The orbitals of the shell with angular momentum `l` in a fixed order (p: px, py, pz, so that the coefficients of a p
orbital are its direction).
"""
shell_types(l) = l == 1 ? Angular[px(), py(), pz()] : l == 2 ? Angular[dxz(), dxy(), dyz(), dx2_y2(), dz2()] :
                 error("No shell for l = $l.")

shell_index(orb) = findfirst(t -> typeof(t) == typeof(orb.type), shell_types(orb.type.l))

"""
    lattice_operations(A, group; tol=1e-2) -> Int64

Number of operations g of `group` that map the lattice with the lattice vectors `A` (columns) onto itself, i.e. for
which A⁻¹ g A is an integer matrix (relative tolerance `tol`, which allows the small distortions of MD cells).
"""
lattice_operations(A, group; tol=1e-2) = count(g -> (M = A \ (g * A); norm(M - round.(M)) < tol * norm(M)), group)

"""
    lattice_point_group_order(A; tol=1e-2) -> Int64

Number of point-group operations of the lattice `A` independent of its orientation (holohedry): the integer matrices M
with entries -1, 0, 1 for which A M A⁻¹ is orthogonal (enough for reduced cells).
"""
function lattice_point_group_order(A; tol=1e-2)
    Ainv = inv(A)
    n = 0
    for entries in Iterators.product(ntuple(_ -> (-1, 0, 1), 9)...)
        M = reshape(collect(Float64, entries), 3, 3)
        abs(abs(det(M)) - 1) < 1e-8 || continue
        g = A * M * Ainv
        norm(g' * g - I) < tol && (n += 1)
    end
    return n
end

"""
    check_symmetry_orientation(lattice, group_name; write_output=true) -> (found, order, other)

Compare the operations of the point group `group_name` that are symmetries of `lattice` (`found`) with the order of
the point group of the lattice itself (`order`) and with the number found for the other group (`other`). Writes a line
to hamster.out and warns if the lattice is more symmetric than the group finds (the structure is not aligned with the
axes of the group: O_h along x, y, z; D_6h with the sixfold axis along z and a twofold axis along x) or if the other
group fits better.
"""
function check_symmetry_orientation(lattice, group_name; write_output=true)
    other_name = group_name == "O_h" ? "D_6h" : "O_h"
    group = point_group(group_name)
    found = lattice_operations(lattice, group)
    other = lattice_operations(lattice, point_group(other_name))
    order = lattice_point_group_order(lattice)
    write_output && append_output_line("enforce_symmetry_group = $group_name: $found of $(length(group)) operations are " *
                                       "symmetries of the lattice (point group of the lattice: $order operations)")
    if found < min(order, length(group))
        @warn "enforce_symmetry_group = $group_name: only $found of its operations are symmetries of the lattice, " *
              "which has $order. The structure is probably not aligned with the axes of the group (O_h: x, y, z; " *
              "D_6h: sixfold axis along z, twofold axis along x)" * (other > found ? "; $other_name fits better ($other)." : ".")
    end
    return found, order, other
end
