"""
    get_index_to_ion_orb_map(Norb) -> Vector{Tuple{Int64, Int64}}

Generate a mapping from basis state indices to corresponding ion and orbital indices.

# Arguments:
- `Norb`: A vector where each element represents the number of orbitals associated with each ion. 

# Returns:
- `index_map`: A vector of tuples where each tuple `(iion, iorb)` represents the ion and orbital index corresponding to `i`.
"""
function get_index_to_ion_orb_map(Norb)
    index_map = Vector{Tuple{Int64, Int64}}(undef, sum(Norb))
    i = 0
    for (iion, n) in enumerate(Norb), jorb in 1:n
        i += 1
        index_map[i] = (iion, jorb)
    end
    return index_map
end

"""
    get_ion_orb_to_index_map(Norb) -> Dict{Tuple{Int64, Int64}, Int64}

Generate a mapping from ion and orbital indices to a basis state index.

# Arguments:
- `Norb`: A vector where each element represents the number of orbitals associated with each ion. 

# Returns:
- `ij_map`: A dictionary where each key is a tuple `(i_ion, i_orb)` representing the ion index `i_ion` and the 
  orbital index `i_orb`. The value corresponding to each key is the basis state index `i`.
"""
function get_ion_orb_to_index_map(Norb)
    ij_map = Dict{Tuple{Int64, Int64}, Int64}()
    for (i, ion_orb) in enumerate(get_index_to_ion_orb_map(Norb))
        ij_map[ion_orb] = i
    end
    return ij_map
end

"""
    get_ion_and_orbital_indices(Norb, index) -> Tuple{Int64, Int64}

Return the ion index `iion` and the orbital index `jorb` corresponding to the basis-state index `index` in the Hamiltonian matrix.

# Arguments:
- `Norb`: A vector where each element represents the number of orbitals for each ion. 
- `index`: The basis-state index for which the corresponding ion and orbital indices are sought.

# Returns:
- `iion`: The index of the ion corresponding to the given `index`.
- `jorb`: The index of the orbital for the ion `iion` corresponding to `index`.
- If `index` is out of bounds, returns `(0, 0)` as a fallback.
"""
function get_ion_and_orbital_indices(Norb, index)
    iion = 0
    cumulative_orb = 0
    
    for orb_count in Norb
        iion += 1
        cumulative_orb += orb_count
        
        if index <= cumulative_orb
            jorb = index - (cumulative_orb - orb_count)
            return iion, jorb
        end
    end
    
    return 0, 0  # Index is out of bounds
end