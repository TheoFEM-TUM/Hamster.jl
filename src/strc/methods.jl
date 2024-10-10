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

"""
    get_nn_thresholds(ions, Ts, point_grid, conf=get_empty_config(); sepNN=get_sepNN(conf))

Calculate nearest-neighbor distance thresholds for pairs of ions based on their positions and types, returning a dictionary of `IonLabel` keys and corresponding minimum distances.

# Arguments
- `ions`: A list of ion objects. Each ion contains its position and type.
- `Ts`: A transformation matrix used for periodic boundary conditions.
- `point_grid`: A grid of points to iterate over for calculating neighbor distances.
- `conf`: A configuration object (optional). Default is `get_empty_config()`.
- `sepNN`: A Boolean flag (optional). If `true`, separate NN thresholds are calculated for each ion pair. If `false`, all ion pairs are treated uniformly. Default is `get_sepNN(conf)`.

# Keyword Arguments
- `sepNN`: Whether to compute separate NN distances for different ion pairs. Default is `get_sepNN(conf)`.

# Returns
- `NN_dict::Dict{IonLabel, Float64}`: A dictionary where each key is an `IonLabel` representing a pair of ion types, and the value is the minimum nearest-neighbor distance for that pair.
"""
function get_nn_thresholds(ions, Ts, point_grid, conf=get_empty_config(); sepNN=get_sepNN(conf))
    NN_dict = Dict{IonLabel, Float64}()
    if !sepNN
        ion_types = get_ion_types(ions, uniq=true)
        for type1 in ion_types, type2 in ion_types
            NN_dict[IonLabel(type1, type2, sorted=false)] = 0.
            NN_dict[IonLabel(type2, type1, sorted=false)] = 0.
        end
    else
        for (iion1, iion2, R) in iterate_nn_grid_points(point_grid)
            ion_label1 = IonLabel(ions[iion1].type, ions[iion2].type, sorted=false)
            ion_label2 = IonLabel(ions[iion1].type, ions[iion2].type, sorted=true)
            Δr = normdiff(ions[iion1].pos, ions[iion2].pos, Ts[:, R])

            for ion_label in [ion_label1, ion_label2]
                if !haskey(NN_dict, ion_label)
                    NN_dict[ion_label] = Δr
                else
                    NN_dict[ion_label] = minimum([Δr, NN_dict[ion_label]])
                end
            end
        end
    end
    return NN_dict
end