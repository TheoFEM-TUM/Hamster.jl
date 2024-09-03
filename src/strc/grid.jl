"""
    PointGrid

A data structure for organizing and managing points in a 3D simulation cell by dividing the space into equally-sized cubic boxes. 

The `PointGrid` structure stores grid points for atom positions, allowing for efficient spatial queries and neighbor searches.

# Fields:
- `grid_size::Float64`: The size of each cubic grid cell in real space.
- `dict0::Dict{SVector{3, Int64}, Vector{Int64}}`: A dictionary that maps grid cell indices to a vector of point indices (e.g., atoms) within that cell.
- `dictR::Dict{SVector{3, Int64}, Vector{Tuple{Int64, Int64}}}`: A dictionary that maps grid cell indices to a vector of tuples. Each tuple contains an atom index and a corresponding replica index, facilitating the handling of periodic boundary conditions.
- `num_points::Int64`: The total number of points (e.g., atoms) managed by the grid.
"""
mutable struct PointGrid
    grid_size :: Float64
    dict0 :: Dict{SVector{3, Int64}, Vector{Int64}}
    dictR :: Dict{SVector{3, Int64}, Vector{Tuple{Int64, Int64}}}
    num_points :: Int64
end

"""
    PointGrid(rs_ion, Ts, conf::TBConfig) -> PointGrid

Constructs a `PointGrid` for managing atomic positions within a 3D simulation cell. 

The function takes atomic positions `rs_ion`, lattice translation vectors `Ts`, and a `TBConfig` configuration object as input, and returns a `PointGrid` object that organizes these atomic positions into a grid of cubic cells.

# Arguments:
- `rs_ion`: A collection (e.g., array) of atomic positions in real space.
- `Ts`: A collection of lattice translation vectors, typically used for defining the periodic boundaries of the simulation cell.
- `conf::Config`: A configuration object containing parameters necessary for setting up the `PointGrid`, such as the grid size.

# Returns:
- `PointGrid`: A `PointGrid` object that partitions the real-space simulation cell into a grid and maps atomic positions to these grid cells.
"""
function PointGrid(rs_ion, Ts, conf=get_empty_config(); grid_size=get_grid_size(conf))
    dict0 = get_grid_dict(rs_ion, grid_size)
    dictR = get_grid_dict(rs_ion, Ts, grid_size)
    return PointGrid(grid_size, dict0, dictR, length(keys(dictR)))
end

"""
    get_grid_dict(rs, grid_size)

Constructs a spatial grid based on the ion positions `rs`, mapping each position to a grid point determined by `grid_size`.

# Arguments
- `rs::AbstractMatrix`: A 3×N matrix where each column represents the position of an ion in 3D space.
- `grid_size::Real`: The size of the grid cells used to map the ion positions.

# Returns
- `grid_dict::Dict{SVector{3, Int64}, Vector{Int64}}`: 
    A dictionary where the keys are grid points (represented as static 3D integer vectors `SVector{3, Int64}`),
    and the values are vectors of ion indices corresponding to the ions located at that grid point.
"""
function get_grid_dict(rs, grid_size)
    grid_dict = Dict{SVector{3, Int64}, Vector{Int64}}()
    @views for i in axes(rs, 2)
        grid_point = SVector{3}(@. Int(floor(rs[:, i] / grid_size)))
        push_grid_point!(grid_dict, grid_point, i)
    end
    return grid_dict
end

"""
    get_grid_dict(rs, Ts, grid_size; δrs=zeros(3, size(rs, 2)))

Constructs a point grid for ion positions `rs`, considering all translation vectors `Ts`.

# Arguments
- `rs::AbstractMatrix`: A 3×N matrix where each column represents the position of an ion in 3D space.
- `Ts::AbstractMatrix`: A 3×M matrix where each column represents a translation vector.
- `grid_size::Real`: The size of the grid cells used to map the ion positions and their translations.

# Returns
- `grid_dict::Dict{SVector{3, Int64}, Vector{Tuple{Int64, Int64}}}`: 
    A dictionary where the keys are grid points (represented as static 3D integer vectors `SVector{3, Int64}`),
    and the values are vectors of tuples `(iion, R)`, where `iion` is the index of the ion and `R` is the index of the translation vector.
"""
function get_grid_dict(rs, Ts, grid_size)
    grid_dict = Dict{SVector{3, Int64}, Vector{Tuple{Int64, Int64}}}()
    @views for R in axes(Ts, 2), iion in axes(rs, 2)
        r⃗ = SVector{3}(rs[:, iion] - Ts[:, R])
        grid_point = get_grid_point(r⃗, grid_size)
        push_grid_point!(grid_dict, grid_point, (iion, R))
    end
    return grid_dict
end

"""
    nn_grid_points(grid_point, grid_dict) -> Vector{SVector{3, Int64}}

Returns a vector of grid points that are nearest neighbors to the given `grid_point`.

This function computes the nearest-neighbor grid points in a 3D grid by considering all adjacent grid cells in the 3x3x3 neighborhood centered around `grid_point`. It checks for the existence of each neighbor in the provided `grid_dict` before including it in the result.

# Arguments:
- `grid_point`: An `SVector{3, Int64}` representing the coordinates of the grid point for which nearest neighbors are sought.
- `grid_dict`: A dictionary (`Dict{SVector{3, Int64}, ...}`) where the keys are grid points in the form of `SVector{3, Int64}`. This dictionary is used to determine if a neighbor grid point exists.

# Returns:
- `Vector{SVector{3, Int64}}`: A vector containing the nearest-neighbor grid points of the specified `grid_point` that are present in the `grid_dict`.
"""
function nn_grid_points(grid_point, grid_dict)
    nn_grid_points = SVector{3, Int64}[]
    for i in -1:1, j in -1:1, k in -1:1
        nn_grid_point = SVector{3}(grid_point .+ [i, j, k])
        if haskey(grid_dict, nn_grid_point)
            push!(nn_grid_points, nn_grid_point)
        end
    end
    return nn_grid_points
end

"""
    push_grid_point!(grid_dict, grid_point, point_info)

Add or update an entry in the grid dictionary with a specified grid point.

# Arguments
- `grid_dict::Dict`: A dictionary where keys are grid points (e.g., `SVector{3, Int64}`) and values are lists of associated points or indices.
- `grid_point`: The key representing the grid point in the dictionary. Typically a static vector or tuple indicating the grid coordinates.
- `point_info`: The value to associate with the `grid_point`. Often this is an index, tuple of indices, or other relevant information.

# Details
If the `grid_point` key already exists in `grid_dict`, the `point_info` is appended to the existing list of values. If the `grid_point` does not exist, a new entry is created with `point_info` as the first element in the list.
"""
function push_grid_point!(grid_dict, grid_point, point_info)
    if haskey(grid_dict, grid_point)
        push!(grid_dict[grid_point], point_info)
    else
        grid_dict[grid_point] = [point_info]
    end
end

"""
    get_grid_point(r⃗, grid_size) -> SVector{3, Int64}

Calculate the grid point corresponding to a given position vector `r⃗` based on a specified grid size.

# Arguments
- `r⃗`: A 3D position vector, typically represented as an `SVector{3, Float64}` or similar type, indicating the coordinates in space.
- `grid_size`: A scalar value representing the size of each grid cell. This is used to determine which grid point the position `r⃗` belongs to.

# Returns
- An `SVector{3, Int64}` representing the grid point. The grid point is computed by dividing each component of `r⃗` by `grid_size`, flooring the result to obtain the integer grid coordinates.
"""
get_grid_point(r⃗, grid_size) = SVector{3}(@. Int(floor(r⃗ / grid_size)))