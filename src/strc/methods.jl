"""
    get_nearest_neighbors(r0, rs, Ts, point_grid::PointGrid; kNN=1, sorted=false)

Find the nearest neighbors (NN) of a given point `r0` within a set of points `rs` and translation vectors `Ts`.

# Arguments
- `r0::AbstractVector{T}`: The reference point in 3D space for which the nearest neighbors are to be found.
- `rs::AbstractMatrix{T}`: A 3xN matrix where each column represents a 3D point in space. These are the potential neighbors.
- `Ts::AbstractMatrix{T}`: A 3xM matrix where each column is a 3D translation vector. These vectors are applied to the points in `rs` to account for periodic boundary conditions.
- `point_grid::PointGrid`: A `PointGrid` structure used for efficient neighbor searching, containing a grid size and a dictionary of grid points.
- `kNN::Int`: The number of nearest neighbors to find. Defaults to 1.

# Returns
- `r_NN::Matrix{Float64}`: A 3xkNN matrix where each column represents the 3D coordinates of one of the `kNN` nearest neighbors to `r0`.

# Description
This function computes the nearest neighbors of a point `r0` by:
1. Locating the grid point corresponding to `r0` in the `point_grid`.
2. Iterating over the neighboring grid points to compute distances to all potential neighbor points in `rs` shifted by translation vectors in `Ts`.
3. Sorting these distances to identify the `kNN` closest neighbors.
4. Returning the positions of these nearest neighbors in a matrix.
"""
function get_nearest_neighbors(r0, rs, Ts, point_grid::PointGrid; kNN=1)
    grid_point = get_grid_point(r0, point_grid.grid_size)
    
    Δrs = Float64[]; inds = Tuple{Int64, Int64}[]
    
    @views for (iion, t) in iterate_nn_grid_points(grid_point, point_grid)
        Δr = normdiff(r0, rs[iion], Ts[:, t])
        push!(Δrs, Δr); push!(inds, (iion, t))
    end

    sorted_inds = sortperm(Δrs)

    r_NN = Vector{SVector{3, Float64}}(undef, kNN)
    @views for k in 1:kNN
        iion, t = inds[sorted_inds][k+1]
        r_NN[k] = SVector{3}(@. rs[iion] - Ts[:, t])
    end
    return r_NN
end