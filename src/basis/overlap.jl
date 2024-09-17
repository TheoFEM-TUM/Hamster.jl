import Base.copy

abstract type MatrixElement end

function get_overlaps(ions, orbitals, conf=get_empty_config())
    overlaps = TBOverlap[]

    unique_ion_types = get_ion_types(ions, uniq=true)

    for iion in eachindex(unique_ion_types), jion in iion:length(unique_ion_types)
        iindex = findnext_ion_of_type(unique_ion_types[iion], ions)
        jindex = findnext_ion_of_type(unique_ion_types[jion], ions)
        interaction_overlaps = get_matrix_element_labels(orbitals[iindex], orbitals[jindex])
        append!(overlaps, interaction_overlaps)
    end

    return overlaps
end

function get_overlap_labels(orbitals_1, orbitals_2)
    interaction_overlaps = TBOverlap[]
    for iorb in eachindex(orbitals_1), jorb in eachindex(orbitals_2)

    end
end

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
    same_ion_label(Vllm, ion_label)

Return `true` if the TB parameter `Vllm` has the ion overlap ion label `ion_label` independently if either is sorted.
"""
function same_ion_label(Vllm::TBOverlap, ion_label::IonLabel)
    return sort(Vllm.ion_label.types) == sort(ion_label.types)
end

(p::TBOverlap)(x...)::Float64 = p.type(x...)
function Base.isequal(param1::TBOverlap, param2::TBOverlap)
    cond1 = typeof(param1.type)==typeof(param2.type)
    cond2 = typeof(param1.orbconfig)==typeof(param2.orbconfig)
    cond3 = param1.ion_label == param2.ion_label
    return cond1 && cond2 && cond3
end

"""
    decide_orbconfig(Vllm, ion_label)

Conjugate the orbital configuration if the sorted ion overlap label of the TB parameter `Vllm` is the conjugate of the
actual ion overlap label `ion_label` of the respective Hamiltonian matrix element.
"""
function decide_orbconfig(Vllm::TBOverlap, ion_label::IonLabel)
    if ion_label == Vllm.ion_label; return Vllm.orbconfig, NormalMode();
    else return conjugate(Vllm.orbconfig), ConjugateMode(); end
end

abstract type OCMode; end
struct NormalMode<:OCMode; end
struct ConjugateMode<:OCMode; end

conjugate(::NormalMode) = ConjugateMode()
conjugate(::ConjugateMode) = NormalMode()

struct ZeroOverlap<:MatrixElement; end
#(me::MatrixElement)(x...) = me(x...)
get_me_label(::Angular, ::Angular, base)::MatrixElement = ZeroOverlap()

function distance_dependence(ME::MatrixElement, orbconfig::OrbitalConfiguration, r, ns, αs)
    R₁ = ndict[ns[1]](αs[1]); R₂ = ndict[ns[2]](αs[2])
    xmin = [-10, -10, -10]; xmax = [15+r/3, 15+r/3, r+15]
    param = stringtype(ME)
    m = mdict_inv[param[3]]
    l₁, l₂ = typeof(orbconfig) == MirrOrb ? (ldict_inv[param[2]], ldict_inv[param[1]]) : (ldict_inv[param[1]], ldict_inv[param[2]])
    sign = l₁ > l₂ ? (-1)^(l₁+l₂) : 1
    Y₁, Y₂ = get_spherical.([l₁, l₂], m)
    f = Overlap(Y₁, R₁, zeros(3), Y₂, R₂, [0., 0., r])
    I, _ = hcubature(f, xmin, xmax, rtol=1e-5, maxevals=1000000, initdiv=5)
    return sign*I
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

"""
    Vssσ
"""
struct Vssσ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(Y₁::s, Y₂::s, base)::MatrixElement = Vssσ(get_base_orb(base))

copy(v::Vssσ) = Vssσ(v.base)

stringtype(::Vssσ) = "ssσ"

(v::Vssσ)(orbconfig, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 0, 0) * 1.

"""
    Vspσ
"""
struct Vspσ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::s, ::pz, base)::MatrixElement = Vspσ(get_base_orb(base))
get_me_label(::pz, ::s, base)::MatrixElement = Vspσ(get_base_orb(base))

copy(v::Vspσ) = Vspσ(v.base)

stringtype(::Vspσ) = "spσ"

(v::Vspσ)(::SymOrb, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 0, 1)*fpz(θ₂, φ₂) - NConst(mode, v.base, 1, 0)*fpz(θ₁, φ₁)
(v::Vspσ)(::DefOrb, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 0, 1)*fpz(θ₂, φ₂)
(v::Vspσ)(::MirrOrb, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = -NConst(mode, v.base, 1, 0)*fpz(θ₁, φ₁)

"""
    Vppσ
"""
struct Vppσ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::pz, ::pz, base)::MatrixElement = Vppσ(get_base_orb(base))

copy(v::Vppσ) = Vppσ(v.base)

stringtype(::Vppσ) = "ppσ"

(v::Vppσ)(orbconfig, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 1)*fpz(θ₁, φ₁) * fpz(θ₂, φ₂)

"""
    Vppπ
"""
struct Vppπ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::px, ::px, base)::MatrixElement = Vppπ(get_base_orb(base))
get_me_label(::py, ::py, base)::MatrixElement = Vppπ(get_base_orb(base))

copy(v::Vppπ) = Vppπ(v.base)

stringtype(::Vppπ) = "ppπ"

(v::Vppπ)(orbconfig, mode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 1)*fpx(θ₁, φ₁)*fpx(θ₂, φ₂) + NConst(mode, v.base, 1, 1)*fpy(θ₁, φ₁)*fpy(θ₂, φ₂)

"""
    Vsdσ
"""
struct Vsdσ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::s, ::dz2, base)::MatrixElement = Vsdσ(get_base_orb(base))
get_me_label(::dz2, ::s, base)::MatrixElement = Vsdσ(get_base_orb(base))

copy(v::Vsdσ) = Vsdσ(v.base)

stringtype(::Vsdσ) = "sdσ"

(v::Vsdσ)(::SymOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 0, 2)*fdz2(v.base[2], θ₂, φ₂) + NConst(mode, v.base, 2, 0)*fdz2(v.base[1], θ₁, φ₁)
(v::Vsdσ)(::SymOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 0, 2)*fdz2(v.base[1], θ₂, φ₂) + NConst(mode, v.base, 2, 0)*fdz2(v.base[2], θ₁, φ₁)

(v::Vsdσ)(::DefOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 0, 2)*fdz2(v.base[2], θ₂, φ₂)
(v::Vsdσ)(::DefOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 0, 2)*fdz2(v.base[1], θ₂, φ₂)

(v::Vsdσ)(::MirrOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 0)*fdz2(v.base[1], θ₁, φ₁)
(v::Vsdσ)(::MirrOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 0)*fdz2(v.base[2], θ₁, φ₁)

"""
    Vpdσ
"""
struct Vpdσ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1,A2}
end

get_me_label(::pz, ::dz2, base)::MatrixElement = Vpdσ(get_base_orb(base))
get_me_label(::dz2, ::pz, base)::MatrixElement = Vpdσ(get_base_orb(base))

copy(v::Vpdσ) = Vpdσ(v.base)

stringtype(::Vpdσ) = "pdσ"

(v::Vpdσ)(::SymOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpz(θ₁, φ₁)*fdz2(v.base[2], θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdz2(v.base[1], θ₁, φ₁)*fpz(θ₂, φ₂)
(v::Vpdσ)(::SymOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpz(θ₁, φ₁)*fdz2(v.base[1], θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdz2(v.base[2], θ₁, φ₁)*fpz(θ₂, φ₂)

(v::Vpdσ)(::DefOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpz(θ₁, φ₁)*fdz2(v.base[2], θ₂, φ₂)
(v::Vpdσ)(::DefOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpz(θ₁, φ₁)*fdz2(v.base[1], θ₂, φ₂)

(v::Vpdσ)(::MirrOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = -NConst(mode, v.base, 2, 1)*fdz2(v.base[1], θ₁, φ₁)*fpz(θ₂, φ₂)
(v::Vpdσ)(::MirrOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = -NConst(mode, v.base, 2, 1)*fdz2(v.base[2], θ₁, φ₁)*fpz(θ₂, φ₂)

"""
    Vpdπ
"""
struct Vpdπ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::px, ::dxz, base)::MatrixElement = Vpdπ(get_base_orb(base))
get_me_label(::dxz, ::px, base)::MatrixElement = Vpdπ(get_base_orb(base))
get_me_label(::py, ::dyz, base)::MatrixElement = Vpdπ(get_base_orb(base))
get_me_label(::dyz, ::py, base)::MatrixElement = Vpdπ(get_base_orb(base))

copy(v::Vpdπ) = Vpdπ(v.base)

stringtype(::Vpdπ) = "pdπ"

(v::Vpdπ)(::SymOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpx(θ₁, φ₁)*fdxz(v.base[2], θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdxz(v.base[1], θ₁, φ₁)*fpx(θ₂, φ₂) + NConst(mode, v.base, 1, 2)*fpy(θ₁, φ₁)*fdyz(v.base[2], θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdyz(v.base[1], θ₁, φ₁)*fpy(θ₂, φ₂)
(v::Vpdπ)(::SymOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpx(θ₁, φ₁)*fdxz(v.base[1], θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdxz(v.base[2], θ₁, φ₁)*fpx(θ₂, φ₂) + NConst(mode, v.base, 1, 2)*fpy(θ₁, φ₁)*fdyz(v.base[1], θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdyz(v.base[2], θ₁, φ₁)*fpy(θ₂, φ₂)

(v::Vpdπ)(::DefOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpx(θ₁, φ₁)*fdxz(v.base[2], θ₂, φ₂) + NConst(mode, v.base, 1, 2)*fpy(θ₁, φ₁)*fdyz(v.base[2], θ₂, φ₂)
(v::Vpdπ)(::DefOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 1, 2)*fpx(θ₁, φ₁)*fdxz(v.base[1], θ₂, φ₂) + NConst(mode, v.base, 1, 2)*fpy(θ₁, φ₁)*fdyz(v.base[1], θ₂, φ₂)

(v::Vpdπ)(::MirrOrb, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = -NConst(mode, v.base, 2, 1)*fdxz(v.base[1], θ₁, φ₁)*fpx(θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdyz(v.base[1], θ₁, φ₁)*fpy(θ₂, φ₂)
(v::Vpdπ)(::MirrOrb, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = -NConst(mode, v.base, 2, 1)*fdxz(v.base[2], θ₁, φ₁)*fpx(θ₂, φ₂) - NConst(mode, v.base, 2, 1)*fdyz(v.base[2], θ₁, φ₁)*fpy(θ₂, φ₂)

"""
    Vddσ

Overlap parameter for overlaps between an s- and a d-orbital with |m|=0.
"""
struct Vddσ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::dz2, ::dz2, base)::MatrixElement = Vddσ(get_base_orb(base))

copy(v::Vddσ) = Vddσ(v.base)

stringtype(::Vddσ) = "ddσ"

(v::Vddσ)(orbconfig, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 2)*fdz2(v.base[1], θ₁, φ₁)*fdz2(v.base[2], θ₂, φ₂)
(v::Vddσ)(orbconfig, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 2)*fdz2(v.base[2], θ₁, φ₁)*fdz2(v.base[1], θ₂, φ₂)

"""
    Vddπ

Overlap parameter for overlaps between two d-orbitals with |m|=1.
"""
struct Vddπ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::dxz, ::dxz, base)::MatrixElement = Vddπ(get_base_orb(base))
get_me_label(::dyz, ::dyz, base)::MatrixElement = Vddπ(get_base_orb(base))

copy(v::Vddπ) = Vddπ(v.base)

stringtype(::Vddπ) = "ddπ"

(v::Vddπ)(orbconfig, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 2)*fdxz(v.base[1], θ₁, φ₁)*fdxz(v.base[2], θ₂, φ₂) + NConst(mode, v.base, 2, 2)*fdyz(v.base[1], θ₁, φ₁)*fdyz(v.base[2], θ₂, φ₂)
(v::Vddπ)(orbconfig, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 2)*fdxz(v.base[2], θ₁, φ₁)*fdxz(v.base[1], θ₂, φ₂) + NConst(mode, v.base, 2, 2)*fdyz(v.base[2], θ₁, φ₁)*fdyz(v.base[1], θ₂, φ₂)

"""
    Vddδ

Overlap parameter for overlaps between two d-orbitals with |m|=2.
"""
struct Vddδ{A1,A2<:Angular}<:MatrixElement
    base :: Tuple{A1, A2}
end

get_me_label(::dxy, ::dxy, base)::MatrixElement = Vddδ(get_base_orb(base))
get_me_label(::dx2_y2, ::dx2_y2, base)::MatrixElement = Vddδ(get_base_orb(base))

copy(v::Vddδ) = Vddδ(v.base)

stringtype(::Vddδ) = "ddδ"

(v::Vddδ)(orbconfig, mode::NormalMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 2)*fdxy(v.base[1], θ₁, φ₁)*fdxy(v.base[2], θ₂, φ₂) + NConst(mode, v.base, 2, 2)*fdx2_y2(v.base[1], θ₁, φ₁)*fdx2_y2(v.base[2], θ₂, φ₂)
(v::Vddδ)(orbconfig, mode::ConjugateMode, θ₁, φ₁, θ₂, φ₂)::Float64 = NConst(mode, v.base, 2, 2)*fdxy(v.base[2], θ₁, φ₁)*fdxy(v.base[1], θ₂, φ₂) + NConst(mode, v.base, 2, 2)*fdx2_y2(v.base[2], θ₁, φ₁)*fdx2_y2(v.base[1], θ₂, φ₂)