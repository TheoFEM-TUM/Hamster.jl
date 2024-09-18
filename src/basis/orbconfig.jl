"""
    OrbitalConfiguration

Abstract type for defining the configuration of orbitals w.r.t. to the overlap label for a specific TB overlap.

Subtypes:
- `SymOrb`: Symmetry in orbitals allows for interchangeable orbital types in labels.
- `DefOrb`: No symmetry; the order of orbital types in labels must be preserved exactly.
- `MirrOrb`: Symmetry requires orbital types to be used in reverse order.
"""
abstract type OrbitalConfiguration; end

"""
    OrbitalConfiguration(Y1, Y2, Ys1, Ys2; sameions=true, ionswap=false) -> OrbitalConfiguration

Determine the `OrbitalConfiguration` type based on the symmetry and relationship between two orbitals, `Y1` and `Y2`, 
and their corresponding sets `Ys1` and `Ys2`. This function decides the appropriate orbital symmetry configuration 
based on the angular momentum and ion types involved.

# Arguments
- `Y1`: The first orbital (expected to have an `l` property for angular momentum).
- `Y2`: The second orbital (expected to have an `l` property for angular momentum).
- `Ys1`: Set of orbitals corresponding to the first ion.
- `Ys2`: Set of orbitals corresponding to the second ion.
- `sameions` (optional, default=true): A boolean indicating whether both orbitals belong to the same ion type.
- `ionswap` (optional, default=false): A boolean indicating whether the ion types are swapped in the configuration.

# Returns:
- Either `SymOrb`, `DefOrb` or `MirrOrb`.
"""
function OrbitalConfiguration(Y1, Y2, Ys1, Ys2; sameions=true, ionswap=false) :: OrbitalConfiguration
    if Y1.l == Y2.l || (Y1 in Ys2 && Y2 in Ys1 && sameions)
        return SymOrb()
    else
        llswap = Y1.l > Y2.l
        if (!ionswap && !llswap) || (ionswap && llswap)
            return DefOrb()
        elseif (ionswap && !llswap) || (llswap && !ionswap)
            return MirrOrb()
        end
    end
end

function OrbitalConfiguration(Y1, Y2, type1, type2, Ys1, Ys2) :: OrbitalConfiguration
    sameions = aresameions(IonLabel(type1, type2))
    ionswap = areswapped(type1, type2)
    return OrbitalConfiguration(Y1, Y2, Ys1, Ys2, sameions=sameions, ionswap=ionswap)
end

"""
    SymOrb()

Symmetry in orbitals: orbital type in label can be interchanged.

# Example
```markdown
<Ga_pσ|As_pσ>

In this case, both orbitals are equal and the label ppσ is symmetric.
```
"""
struct SymOrb<:OrbitalConfiguration; end

"""
    DefOrb()

No symmetry in orbitals: orbital type in label has to be used in the same order as given.
"""
struct DefOrb<:OrbitalConfiguration; end

"""
    MirrOrb()

No symmetry in orbitals: orbital type in label has to be used in reverse order as given.
"""
struct MirrOrb<:OrbitalConfiguration; end

conjugate(oc::SymOrb) = oc
conjugate(::DefOrb) = MirrOrb()
conjugate(::MirrOrb) = DefOrb()

abstract type OCMode; end
struct NormalMode<:OCMode; end
struct ConjugateMode<:OCMode; end

conjugate(::NormalMode) = ConjugateMode()
conjugate(::ConjugateMode) = NormalMode()
