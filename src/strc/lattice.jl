"""
    get_translation_vectors(rs_ion, lattice, conf=get_empty_config(); Rmax=get_Rmax(lattice, conf), rcut=get_rcut(conf))

Generate a set of translation vectors based on ion positions `rs_ion`, the lattice vectors `lattice`, and optional configuration settings. The function returns the translation vectors that satisfy the maximum interatomic distance `rcut`, which is determined from the configuration or can be manually specified.

# Arguments
- `rs_ion::AbstractMatrix`: Matrix of ion positions in fractional coordinates. Each column represents the position of an ion.
- `lattice::AbstractMatrix`: Lattice vectors of the system, where each column represents a lattice vector.
- `conf::Config`: (Optional) Configuration object containing settings like `Rmax` and `rcut`. Defaults to an empty configuration.
- `Rmax::Int64`: (Optional) Maximum entry magnitude for the initial set of translation vectors. Defaults to the value from the configuration.
- `rcut::Float64`: (Optional) Maximum interatomic distance. If set to 0, all initial translation vectors will be returned. Defaults to the value from the configuration.

# Returns
- `AbstractMatrix`: A matrix containing the translation vectors as columns. The vectors are filtered based on the `rcut` value to include only those within the specified interatomic distance.
"""
function get_translation_vectors(rs_ion, lattice, conf=get_empty_config(); Rmax=get_Rmax(lattice, conf), rcut=get_rcut(conf))
    # Get an initial set of lattice translation vectors
    Rs = get_translation_vectors(Rmax)
    if rcut == 0.; return Rs; end
    Ts = frac_to_cart(Rs, lattice)
    
    point_grid = PointGrid(rs_ion, Ts, grid_size=rcut)
    
    R_inds = Int64[]
    for (iion1, iion2, R) in iterate_nn_grid_points(point_grid)
        Δr = normdiff(rs_ion[:, iion1], rs_ion[:, iion2], Ts[:, R])
        if Δr ≤ rcut; push!(R_inds, R); end
    end
    return Rs[:, unique(R_inds)]
end

"""
    get_translation_vectors(M::Int64)

Generate a set of translation vectors within a cubic grid defined by the maximum entry magnitude `M`.

# Arguments
- `M::Int64`: The maximum magnitude of the entries in the translation vectors. The resulting grid will include all integer vectors where each component is in the range `-M` to `M`.

# Returns
- `Rs::Matrix{Float64}`: A 3xN matrix where `N` is the total number of translation vectors. Each column of `Rs` represents a translation vector with components within the specified magnitude range.
"""
function get_translation_vectors(M::Int64)
    NR = (2*M + 1)^3; Rs = zeros(Float64, 3, NR)
    tls = [(i, j, k) for i in -M:M for j in -M:M for k in -M:M]
    for R in 1:NR
        Rs[:, R] .= tls[R]
    end
    return Rs
end

"""
    get_Rmax(lattice, conf=get_empty_config(); rcut=get_rcut(conf), Rmax=get_Rmax(conf), upperR=10)

Determines the maximum radius `Rmax` for the given lattice and configuration parameters based on the cutoff radius `rcut`.

# Arguments
- `lattice`: The lattice parameters used to convert fractional to Cartesian coordinates.
- `conf`: (Optional) Configuration object. Defaults to `get_empty_config()`.
- `rcut`: (Optional) The cutoff radius used to determine the maximum radius. Defaults to `get_rcut(conf)`.
- `Rmax`: (Optional) Initial guess for the maximum radius. Defaults to `get_Rmax(conf)`.
- `upperR`: (Optional) Upper limit for the search radius. Defaults to 10.

# Returns
- `Rmax::Int`: The maximum radius such that the spherical region of radius `Rmax` contains all points within the cutoff radius `rcut`.
"""
function get_Rmax(lattice, conf=get_empty_config(); rcut=get_rcut(conf), Rmax=get_Rmax(conf), upperR=10)
    if rcut == 0.
        return Rmax
    else    
        for R in 1:maximum([Rmax, upperR])
            if norm(frac_to_cart([R, R, R], lattice)) ≥ rcut; return R+1 ; end
        end
        throw("Could not find suitable Rmax. Maybe decrease rcut?")
    end
end

"""
    findR0(Rs::Matrix{<:Number})

Finds the index of the translation vector `[0, 0, 0]` in the matrix `Rs`.

# Arguments
- `Rs::Matrix{<:Number}`: A 3xN matrix where each column represents a translation vector.

# Returns
- `R::Int`: The index of the column in `Rs` that contains the vector `[0, 0, 0]`. If the vector `[0, 0, 0]` is not found, returns `0`.
"""
function findR0(Rs)
    R = findfirst(isequal([0, 0, 0]), eachcol(Rs))
    return R === nothing ? 0 : R
end