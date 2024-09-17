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

"""
    OrbitalConfiguration(Y1, Y2, type1, type2, Ys1, Ys2) :: OrbitalConfiguration

Determine and return the appropriate `OrbitalConfiguration` for the tight-binding overlap between orbitals `Y1` and `Y2`.

# Arguments:
- `Y1`::Angular: Orbital information for the first orbital.
- `Y2`::Angular: Orbital information for the second orbital.
- `type1`: Ion type or identifier for the first ion involved in the overlap.
- `type2`: Ion type or identifier for the second ion involved in the overlap.
- `Ys1`: A set of orbital configurations associated with the first ion.
- `Ys2`: A set of orbital configurations associated with the second ion.

# Returns:
- Either `SymOrb`, `DefOrb` or `MirrOrb`.
"""
function OrbitalConfiguration(Y1, Y2, type1, type2, Ys1, Ys2) :: OrbitalConfiguration
    if Y1.l == Y2.l || (Y1 in Ys2 && Y2 in Ys1 && type1 == type2)
        return SymOrb()
    else
        llswap = Y1.l > Y2.l
        ionswap = IonLabel(type1, type2) ≠ IonLabel(type1, type2, sorted=false)
        if (!ionswap && !llswap) || (ionswap && llswap)
            return DefOrb()
        elseif (ionswap && !llswap) || (llswap && !ionswap)
            return MirrOrb()
        end
    end
end
