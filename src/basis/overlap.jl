import Base.copy

abstract type MatrixElement end

"""
    TBOverlap{T<:MatrixElement, OC<:OrbitalConfiguration}

Represents a tight-binding (TB) overlap used in calculations, characterized by its type, orbital configuration, and ion label. This represents one interaction term.

# Fields
- `type::T`: Specifies the type of the TB overlap parameter, where `T` is a subtype of `MatrixElement`. This field indicates the specific kind of matrix element or overlap parameter.
- `orbconfig::OC`: Represents the `OrbitalConfiguration` associated with the TB overlap parameter.
- `ion_label::IonLabel`: Labels the ions that are involved in the interaction term.
"""
struct TBOverlap{T<:MatrixElement, OC<:OrbitalConfiguration}
    type :: T
    orbconfig :: OC
    ion_label :: IonLabel
end

"""
    Base.isequal(ov1::TBOverlap, ov2::TBOverlap)

Compare two `TBOverlap` objects, `ov1` and `ov2`, for equality.

# Arguments
- `ov1::TBOverlap`: The first `TBOverlap` instance.
- `ov2::TBOverlap`: The second `TBOverlap` instance.

# Returns
- `Bool`: `true` if the two `TBOverlap` instances are considered equal, `false` otherwise.
"""
function Base.isequal(ov1::OV1, ov2::OV2) where {OV1,OV2<:TBOverlap}
    cond1 = typeof(ov1.type) == typeof(ov2.type)
    cond2 = typeof(ov1.orbconfig) == typeof(ov2.orbconfig)
    cond3 = ov1.ion_label == ov2.ion_label
    return cond1 && cond2 && cond3
end

Base.hash(a::TBOverlap, h::UInt) = hash((a.type, a.orbconfig, a.ion_label), h)

"""
    get_overlaps(ions, orbitals, conf=get_empty_config())

Compute all tight-binding (TB) overlaps between pairs of orbitals for a given list of ions.

# Arguments
- `ions`: A vector of ions from which overlaps are to be computed.
- `orbitals`: A vector of orbitals corresponding to each ion in `ions`.
- `conf`: Optional configuration parameter (default: `get_empty_config()`).

# Returns
- `Vector{TBOverlap}`: A vector containing all computed `TBOverlap` instances for the ion and orbital pairs.
"""
function get_overlaps(ions, orbitals, conf=get_empty_config())
    overlaps = TBOverlap[]

    unique_ion_types = get_ion_types(ions, uniq=true)
    for iion in eachindex(unique_ion_types), jion in iion:length(unique_ion_types)
        ion_label = IonLabel(unique_ion_types[iion], unique_ion_types[jion])
        ionswap = areswapped(unique_ion_types[iion], unique_ion_types[jion])
        iindex = findnext_ion_of_type(unique_ion_types[iion], ions)
        jindex = findnext_ion_of_type(unique_ion_types[jion], ions)
        interaction_overlaps = get_overlaps_for_orbitals(orbitals[iindex], orbitals[jindex], ion_label, ionswap)
        append!(overlaps, interaction_overlaps)
    end

    return overlaps
end

"""
    get_overlaps_for_orbitals(orbitals_1, orbitals_2, ion_label, ionswap)

Compute the list of tight-binding (TB) overlaps for pairs of orbitals from two ions (`orbitals_1` and `orbitals_2`) based on the given `ion_label` and whether the ions are swapped (`ionswap`).

# Arguments
- `orbitals_1`: A collection of orbitals from the first ion.
- `orbitals_2`: A collection of orbitals from the second ion.
- `ion_label`: An `IonLabel` instance representing the ion types involved in the interaction.
- `ionswap`: A boolean flag indicating if the ions are swapped in the interaction.

# Returns
- `Vector{TBOverlap}`: A unique list of `TBOverlap` instances describing the possible overlaps between the orbitals of the two ions, taking into account symmetry and ion swaps.
"""
function get_overlaps_for_orbitals(orbitals_1, orbitals_2, ion_label, ionswap)
    sameions = aresameions(ion_label)
    interaction_overlaps = TBOverlap[]
    
    for (iorb, orb_i) in enumerate(orbitals_1), orb_j in orbitals_2[ifelse(sameions, iorb, 1):end]
        Ys_i = get_orbital_list(orb_i.type)
        Ys_j = get_orbital_list(orb_j.type)
        base_ls = ionswap ? Tuple([orb_j.type.l, orb_i.type.l]) : Tuple([orb_i.type.l, orb_j.type.l])
        for Y_i in Ys_i, Y_j in Ys_j
            me_label = get_me_label(Y_i, Y_j, base_ls)
            if !(typeof(me_label) <: ZeroOverlap)
                orbconfig = OrbitalConfiguration(Y_i, Y_j, Ys_i, Ys_j, sameions=sameions, ionswap=ionswap)
                push!(interaction_overlaps, TBOverlap(me_label, orbconfig, ion_label))
            end
        end
    end
    return unique(interaction_overlaps)
end

"""
    Base.string(p::TBOverlap; apply_oc=false)::String

Converts a `TBOverlap` instance into a string, e.g., `"Ga+As_ssσ"``. If `apply_oc=true`, the `OrbitalConfiguration` is taken into account. 
This is only relevant for `orbconfig<:MirrOrb`, where the order of orbitals in the label is inverted, e.g., the overlap `spσ`` with `orbconfig<:MirrOrb` would become `psσ`.

# Arguments
- `p::TBOverlap`: The `TBOverlap` instance to be converted into a string.
- `apply_oc::Bool`: A flag indicating whether to apply orbital configuration-specific formatting.
"""
function Base.string(p::TBOverlap; apply_oc=false)::String
    if apply_oc == false || !(typeof(p.orbconfig) <: MirrOrb)
        return string(p.ion_label)*"_"*string(p.type)
    elseif apply_oc == true && typeof(p.orbconfig) <: MirrOrb
        vllm = string(p.type)
        return string(p.ion_label)*"_"*vllm[2]*vllm[1]*vllm[3]
    end
end

"""
    me_to_overlap_label(me::MatrixElement) -> SVector{3, Int64}

Converts the matrix element `me` to an overlap label with l, l', m as a static array.
"""
me_to_overlap_label(me) = string_to_overlap_label(string(me))

"""
    get_baseorb_ls(overlap::TBOverlap) -> Tuple{Int, Int}

Extract the orbital angular momentum quantum numbers `l` for the two orbitals in the `baseorb` field of 
the given `TBOverlap` object.

# Arguments
- `overlap::TBOverlap`: A `TBOverlap` object representing the overlap between two orbitals in a tight-binding model. 
  The `baseorb` field contains information about the two orbitals.

# Returns
- A tuple `(l1, l2)` where `l1` is the angular momentum quantum number of the first orbital and 
  `l2` is that of the second orbital.
"""
function get_baseorb_ls(overlap::TBOverlap)
    return overlap.type.base_ls[1], overlap.type.base_ls[2]
end

"""
    same_ion_label(Vllm, ion_label)

Return `true` if the TB overlap `Vllm` has the ion overlap ion label `ion_label` independently if either is sorted.
"""
function same_ion_label(Vllm::TBOverlap, ion_label::IonLabel)
    return sort(Vllm.ion_label.types) == sort(ion_label.types)
end

(p::TBOverlap)(x...)::Float64 = p.type(x...)

struct ZeroOverlap<:MatrixElement; end
#(me::MatrixElement)(x...) = me(x...)
get_me_label(::Angular, ::Angular, base_ls)::MatrixElement = ZeroOverlap()

"""
    decide_orbconfig(Vllm, ion_label)

Conjugate the orbital configuration if the sorted ion overlap label of the TB parameter `Vllm` is the conjugate of the
actual ion overlap label `ion_label` of the respective Hamiltonian matrix element.
"""
function decide_orbconfig(Vllm::TBOverlap, ion_label::IonLabel)::Tuple{Union{SymOrb, MirrOrb, DefOrb}, Union{NormalMode, ConjugateMode}}
    if ion_label == Vllm.ion_label; return Vllm.orbconfig, NormalMode();
    else return conjugate(Vllm.orbconfig), ConjugateMode(); end
end

"""
    NConst(baseorb, l₁, l₂)

Determine the normalization constant for the matrix element between orbitals in `baseorb` with respective
components `l₁` and `l₂`.
"""
function NConst(::NormalMode, base, l₁, l₂)
    Nspd₁ = Nspd(base[1]); Nspd₂ = Nspd(base[2])
    return √(Nspd₁[l₁+1]*Nspd₂[l₂+1] / (sum(Nspd₁)*sum(Nspd₂)))
end

function NConst(::ConjugateMode, base, l₁, l₂)
    Nspd₁ = Nspd(base[2]); Nspd₂ = Nspd(base[1])
    return √(Nspd₁[l₁+1]*Nspd₂[l₂+1] / (sum(Nspd₁)*sum(Nspd₂)))
end

function phase(base)
    if base[1].l ≥ 0 && base[2].l ≥ 0
        return +1
    else
        return -1
    end
end

"""
    Vssσ
"""
struct Vssσ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(Y₁::s, Y₂::s, base_ls)::MatrixElement = Vssσ(base_ls)

copy(v::Vssσ) = Vssσ(v.base_ls)

Base.string(::Vssσ) = "ssσ"

(v::Vssσ)(base, orbconfig, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 0, 0) * 1.

"""
    Vspσ
"""
struct Vspσ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::s, ::pz, base_ls)::MatrixElement = Vspσ(base_ls)
get_me_label(::pz, ::s, base_ls)::MatrixElement = Vspσ(base_ls)

copy(v::Vspσ) = Vspσ(v.base_ls)

Base.string(::Vspσ) = "spσ"

(v::Vspσ)(base, ::SymOrb, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 0, 1)*fpz(θ₂, φ₂) + phase(base)*NConst(mode, base, 1, 0)*fpz(θ₁, φ₁)
(v::Vspσ)(base, ::DefOrb, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 0, 1)*fpz(θ₂, φ₂)
(v::Vspσ)(base, ::MirrOrb, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = phase(base)*NConst(mode, base, 1, 0)*fpz(θ₁, φ₁)

"""
    Vppσ
"""
struct Vppσ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::pz, ::pz, base_ls)::MatrixElement = Vppσ(base_ls)

copy(v::Vppσ) = Vppσ(v.base_ls)

Base.string(::Vppσ) = "ppσ"

(v::Vppσ)(base, orbconfig, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 1)*fpz(θ₁, φ₁) * fpz(θ₂, φ₂)

"""
    Vppπ
"""
struct Vppπ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::px, ::px, base_ls)::MatrixElement = Vppπ(base_ls)
get_me_label(::py, ::py, base_ls)::MatrixElement = Vppπ(base_ls)

copy(v::Vppπ) = Vppπ(v.base_ls)

Base.string(::Vppπ) = "ppπ"

(v::Vppπ)(base, orbconfig, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 1)*fpx(θ₁, φ₁)*fpx(θ₂, φ₂) + NConst(mode, base, 1, 1)*fpy(θ₁, φ₁)*fpy(θ₂, φ₂)

"""
    Vsdσ
"""
struct Vsdσ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::s, ::dz2, base_ls)::MatrixElement = Vsdσ(base_ls)
get_me_label(::dz2, ::s, base_ls)::MatrixElement = Vsdσ(base_ls)

copy(v::Vsdσ) = Vsdσ(v.base_ls)

Base.string(::Vsdσ) = "sdσ"

(v::Vsdσ)(base, ::SymOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 0, 2)*fdz2(base[2], θ₂, φ₂) + NConst(mode, base, 2, 0)*fdz2(base[1], θ₁, φ₁)
(v::Vsdσ)(base, ::SymOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 0, 2)*fdz2(base[1], θ₂, φ₂) + NConst(mode, base, 2, 0)*fdz2(base[2], θ₁, φ₁)

(v::Vsdσ)(base, ::DefOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 0, 2)*fdz2(base[2], θ₂, φ₂)
(v::Vsdσ)(base, ::DefOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 0, 2)*fdz2(base[1], θ₂, φ₂)

(v::Vsdσ)(base, ::MirrOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 0)*fdz2(base[1], θ₁, φ₁)
(v::Vsdσ)(base, ::MirrOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 0)*fdz2(base[2], θ₁, φ₁)

"""
    Vpdσ
"""
struct Vpdσ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::pz, ::dz2, base_ls)::MatrixElement = Vpdσ(base_ls)
get_me_label(::dz2, ::pz, base_ls)::MatrixElement = Vpdσ(base_ls)

copy(v::Vpdσ) = Vpdσ(v.base_ls)

Base.string(::Vpdσ) = "pdσ"

(v::Vpdσ)(base, ::SymOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpz(θ₁, φ₁)*fdz2(base[2], θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdz2(base[1], θ₁, φ₁)*fpz(θ₂, φ₂)
(v::Vpdσ)(base, ::SymOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpz(θ₁, φ₁)*fdz2(base[1], θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdz2(base[2], θ₁, φ₁)*fpz(θ₂, φ₂)

(v::Vpdσ)(base, ::DefOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpz(θ₁, φ₁)*fdz2(base[2], θ₂, φ₂)
(v::Vpdσ)(base, ::DefOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpz(θ₁, φ₁)*fdz2(base[1], θ₂, φ₂)

(v::Vpdσ)(base, ::MirrOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = phase(base)*NConst(mode, base, 2, 1)*fdz2(base[1], θ₁, φ₁)*fpz(θ₂, φ₂)
(v::Vpdσ)(base, ::MirrOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = phase(base)*NConst(mode, base, 2, 1)*fdz2(base[2], θ₁, φ₁)*fpz(θ₂, φ₂)

"""
    Vpdπ
"""
struct Vpdπ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::px, ::dxz, base_ls)::MatrixElement = Vpdπ(base_ls)
get_me_label(::dxz, ::px, base_ls)::MatrixElement = Vpdπ(base_ls)
get_me_label(::py, ::dyz, base_ls)::MatrixElement = Vpdπ(base_ls)
get_me_label(::dyz, ::py, base_ls)::MatrixElement = Vpdπ(base_ls)

copy(v::Vpdπ) = Vpdπ(v.base_ls)

Base.string(::Vpdπ) = "pdπ"

(v::Vpdπ)(base, ::SymOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpx(θ₁, φ₁)*fdxz(base[2], θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdxz(base[1], θ₁, φ₁)*fpx(θ₂, φ₂) + NConst(mode, base, 1, 2)*fpy(θ₁, φ₁)*fdyz(base[2], θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdyz(base[1], θ₁, φ₁)*fpy(θ₂, φ₂)
(v::Vpdπ)(base, ::SymOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpx(θ₁, φ₁)*fdxz(base[1], θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdxz(base[2], θ₁, φ₁)*fpx(θ₂, φ₂) + NConst(mode, base, 1, 2)*fpy(θ₁, φ₁)*fdyz(base[1], θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdyz(base[2], θ₁, φ₁)*fpy(θ₂, φ₂)

(v::Vpdπ)(base, ::DefOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpx(θ₁, φ₁)*fdxz(base[2], θ₂, φ₂) + NConst(mode, base, 1, 2)*fpy(θ₁, φ₁)*fdyz(base[2], θ₂, φ₂)
(v::Vpdπ)(base, ::DefOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 1, 2)*fpx(θ₁, φ₁)*fdxz(base[1], θ₂, φ₂) + NConst(mode, base, 1, 2)*fpy(θ₁, φ₁)*fdyz(base[1], θ₂, φ₂)

(v::Vpdπ)(base, ::MirrOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = phase(base)*NConst(mode, base, 2, 1)*fdxz(base[1], θ₁, φ₁)*fpx(θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdyz(base[1], θ₁, φ₁)*fpy(θ₂, φ₂)
(v::Vpdπ)(base, ::MirrOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = phase(base)*NConst(mode, base, 2, 1)*fdxz(base[2], θ₁, φ₁)*fpx(θ₂, φ₂) + phase(base)*NConst(mode, base, 2, 1)*fdyz(base[2], θ₁, φ₁)*fpy(θ₂, φ₂)

"""
    Vddσ

Overlap parameter for overlaps between an s- and a d-orbital with |m|=0.
"""
struct Vddσ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::dz2, ::dz2, base_ls)::MatrixElement = Vddσ(base_ls)

copy(v::Vddσ) = Vddσ(v.base_ls)

Base.string(::Vddσ) = "ddσ"

(v::Vddσ)(base, orbconfig, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 2)*fdz2(base[1], θ₁, φ₁)*fdz2(base[2], θ₂, φ₂)
(v::Vddσ)(base, orbconfig, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 2)*fdz2(base[2], θ₁, φ₁)*fdz2(base[1], θ₂, φ₂)

"""
    Vddπ

Overlap parameter for overlaps between two d-orbitals with |m|=1.
"""
struct Vddπ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::dxz, ::dxz, base_ls)::MatrixElement = Vddπ(base_ls)
get_me_label(::dyz, ::dyz, base_ls)::MatrixElement = Vddπ(base_ls)

copy(v::Vddπ) = Vddπ(v.base_ls)

Base.string(::Vddπ) = "ddπ"

(v::Vddπ)(base, orbconfig, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 2)*fdxz(base[1], θ₁, φ₁)*fdxz(base[2], θ₂, φ₂) + NConst(mode, base, 2, 2)*fdyz(base[1], θ₁, φ₁)*fdyz(base[2], θ₂, φ₂)
(v::Vddπ)(base, orbconfig, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 2)*fdxz(base[2], θ₁, φ₁)*fdxz(base[1], θ₂, φ₂) + NConst(mode, base, 2, 2)*fdyz(base[2], θ₁, φ₁)*fdyz(base[1], θ₂, φ₂)

"""
    Vddδ

Overlap parameter for overlaps between two d-orbitals with |m|=2.
"""
struct Vddδ<:MatrixElement
    base_ls :: Tuple{Int64, Int64}
end

get_me_label(::dxy, ::dxy, base_ls)::MatrixElement = Vddδ(base_ls)
get_me_label(::dx2_y2, ::dx2_y2, base_ls)::MatrixElement = Vddδ(base_ls)

copy(v::Vddδ) = Vddδ(v.base_ls)

Base.string(::Vddδ) = "ddδ"

(v::Vddδ)(base, orbconfig, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 2)*fdxy(base[1], θ₁, φ₁)*fdxy(base[2], θ₂, φ₂) + NConst(mode, base, 2, 2)*fdx2_y2(base[1], θ₁, φ₁)*fdx2_y2(base[2], θ₂, φ₂)
(v::Vddδ)(base, orbconfig, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, base, 2, 2)*fdxy(base[2], θ₁, φ₁)*fdxy(base[1], θ₂, φ₂) + NConst(mode, base, 2, 2)*fdx2_y2(base[2], θ₁, φ₁)*fdx2_y2(base[1], θ₂, φ₂)