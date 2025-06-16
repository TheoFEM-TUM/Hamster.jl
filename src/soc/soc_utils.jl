"""
    num_mdict::Dict{String, Array{Int}}

A dictionary mapping the orbital type to an array of `m` quantum numbers.
The keys are orbital types as strings ("s", "p", "d") and the values are arrays
containing the possible `m` values for each orbital type.
"""
#TODO: Check whether this is already implemented elsewhere in the package
const num_mdict = Dict("s" => [0], "p" => [-1, 0, 1], "d" => [-2, -1, 0, 1, 2])

"""
    create_basis_lm(orb::String) -> Vector{Vector{Int64}}

Create the basis set in the |lm,s> representation for a given type of orbital specified by `orb`.
This function asserts that `orb` is one of the strings "s", "p", or "d", which correspond to 
the different orbital types. It uses the `num_mdict` to fetch the corresponding `m` values and
constructs a basis set with `l`, `m`, and `spin` quantum numbers.

# Arguments
- `orb`: A string specifying the orbital type ("s", "p", or "d").

# Returns
- `basis`: A vector of vectors, with each sub-vector containing the quantum numbers `[l, m, spin]`.
"""
function get_lms_basis(orb::String)
    @assert orb in ["s", "p", "d"] "The orb parameter must be one of the s, p, d values in the format of String"
    
    # Directly define the mapping inside the function
    l_value = findfirst(x -> x == orb, ["s", "p", "d"]) - 1
    mlist = num_mdict[orb]
    basis = Vector{Vector{Int64}}()
    
    for m in mlist
        for spin in [-1, 1]
            push!(basis, Vector{Int64}([l_value, m, spin]))  # Ensure the pushed vector is of type Vector{Int64}
        end
    end
    return basis
end

"""
    mapLpSm(lms::Vector{Int}) -> Tuple{Float64, Vector{Int}}

Map the |lm,s> state to another state with the momentum and spin ladder operator L+S-.
The function calculates the coefficient for the operation and the resulting state.

# Arguments
- `lms`: A vector of quantum numbers `[l, m, s]`.

# Returns
- A tuple of the coefficient of the operation and the new quantum numbers `[l, m+1, s-2]`.
"""
function mapLpSm(lms::Vector{Int})
    l, m, s = lms
    cof = 0.0  # Initialize
    if s == 1
        cof = sqrt((l - m) * (l + m + 1))
    elseif s == -1
        cof = 0.0
    end
    return cof, [l, m + 1, s - 2]
end

"""
    mapLmSp(lms::Vector{Int}) -> Tuple{Float64, Vector{Int}}

Map the |lm,s> state to another state with the momentum and spin ladder operator L-S+.
The function calculates the coefficient for the operation and the resulting state.

# Arguments
- `lms`: A vector of quantum numbers `[l, m, s]`.

# Returns
- A tuple of the coefficient of the operation and the new quantum numbers `[l, m-1, s+2]`.
"""
function mapLmSp(lms::Vector{Int})
    l, m, s = lms
    cof = 0.0  # Initialize
    if s == 1
        cof = 0.0
    elseif s == -1
        cof = sqrt((l + m) * (l - m + 1))
    end
    return cof, [l, m - 1, s + 2]
end

"""
    mapLzSz(lms::Vector{Int}) -> Tuple{Int, Vector{Int}}

Map the |lm,s> state to itself while calculating the coefficient for the LzSz operation.

# Arguments
- `lms`: A vector of quantum numbers `[l, m, s]`.

# Returns
- A tuple of the coefficient m for s=1 or -m for s=-1, and the unchanged quantum numbers `[l, m, s]`.
"""
function mapLzSz(lms::Vector{Int})
    l, m, s = lms
    cof = m * s / 2
    return cof, [l, m, s]
end

"""
    get_matrix_lmbasis(basis::Vector{Vector{Int}}) -> Matrix{Complex{Float64}}

Construct the spin-orbit coupling matrix in the |lm,s> basis.
This matrix represents the dot product of the angular momentum operators with the spin operators.

# Arguments
- `basis`: A vector of basis sets, each represented by a vector of quantum numbers `[l, m, spin]`.

# Returns
- A matrix of type `Complex{Float64}` representing the spin-orbit coupling in the given basis.
"""
function get_matrix_lmbasis(basis::Vector{Vector{Int}})
    ndim = length(basis)
    MatLpSm = zeros(Float64, ndim, ndim)
    MatLmSp = zeros(Float64, ndim, ndim)
    MatLzSz = zeros(Float64, ndim, ndim)

    # Pre-compute a map for basis to index
    basis_map = Dict(bas => i for (i, bas) in enumerate(basis))

    for i in 1:ndim
        row = i
        for (map_func, Mat) in [(mapLpSm, MatLpSm), (mapLmSp, MatLmSp), (mapLzSz, MatLzSz)]
            cof, bas = map_func(basis[i])
            col = get(basis_map, bas, nothing)
            if col !== nothing
                Mat[row, col] = cof
            end
        end
    end
    
    LdotS = 0.5 * (MatLpSm + MatLmSp) + MatLzSz #TODO: is this even correct?
    return Complex{Float64}.(LdotS)
end

"""
    thresholding(c::Complex, real_threshold::Real=1e-15) -> Complex

Apply thresholding to a complex number, setting real and imaginary parts below the threshold to zero.

# Arguments
- `c::Complex`: The complex number to threshold.
- `real_threshold::Real`: The threshold value for the real and imaginary parts (default: 1e-15).

# Returns
- `Complex`: The thresholded complex number.
"""
function thresholding(c::Complex, real_threshold::Real=1e-15, imag_threshold::Real=real_threshold)
    _real = real_threshold < abs(real(c)) ? real(c) : 0.0
    _imag = imag_threshold < abs(imag(c)) ? imag(c) : 0.0
    return Complex(_real, _imag)
end

"""
    separate_spin_secs(orb_order::Vector{String}) -> Vector{String}

Reorder the basis by separating the spin components. This function does not alter 
the order of spatial dimensions (e.g., s, py, sp3₁), but sorts the spin sector such 
that all up-spin (↑) components appear before all down-spin (↓) components.
"""
function separate_spin_secs(orb_order)
        up_spin = filter(x -> occursin("↑", x), orb_order)
        down_spin = filter(x -> occursin("↓", x), orb_order)
    return vcat(up_spin, down_spin)
end 

"""
    convert_block_matrix_to_sparse(M::BlockDiagonal; sp_tol=1e-10) -> SparseMatrixCSC

Converts a `BlockDiagonal` matrix `M` to a `SparseMatrixCSC` by iterating through
each block and collecting the nonzero entries (within a given tolerance).

# Arguments
- `M::BlockDiagonal`: A block diagonal matrix, typically from `BlockDiagonals.jl` or similar libraries.
"""
function convert_block_matrix_to_sparse(M::BlockDiagonal; sp_tol=1e-10)
    is = Int64[]; js = Int64[]; vals = ComplexF64[]
    sizelist = blocksizes(M)
    i_cum_sizelist = [sum([s[1] for s in sizelist[1:block_idx-1]]) for block_idx in eachindex(blocks(M))]
    j_cum_sizelist = [sum([s[2] for s in sizelist[1:block_idx-1]]) for block_idx in eachindex(blocks(M))]
    @views for (block_idx, block) in enumerate(blocks(M))
        block_size = sizelist[block_idx]
        for block_j in 1:block_size[2], block_i in 1:block_size[1]
            val = block[block_i, block_j]
            i = block_idx > 1 ? i_cum_sizelist[block_idx] + block_i : block_i
            j = block_idx > 1 ? j_cum_sizelist[block_idx] + block_j : block_j
            if abs(val) ≥ sp_tol
                push!(is, i)
                push!(js, j)
                push!(vals, val)
            end
        end
    end
    M_sparse = sparse(is, js, vals, size(M, 1), size(M, 2))
    return M_sparse
end