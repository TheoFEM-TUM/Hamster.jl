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
        r_NN[k] = SVector{3}(apply_pbc(rs[iion], Ts[:, t]))
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
                if !haskey(NN_dict, ion_label) && Δr ≠ 0
                    NN_dict[ion_label] = Δr
                elseif Δr ≠ 0
                    NN_dict[ion_label] = minimum([Δr, NN_dict[ion_label]])
                end
            end
        end
    end
    return NN_dict
end

"""
    get_nn_cutoffs(ions, Ts, point_grid, conf=get_empty_config(); sepNN=get_sepNN(conf), shell_gap=1.1, bin_width=0.01)

Distance up to which a pair of ions counts as nearest neighbors (label 1 with `sepNN = true`), for every `IonLabel`:
the midpoint between the first and the second neighbor shell of the reference positions `ion.pos`. Distorted
structures (MD snapshots, a distorted POSCAR) therefore keep the labels of their bonds as long as no bond is stretched
or compressed past the midpoint; a bond that crosses it changes its parameters abruptly.

The shells are found from the sorted pair distances: the first shell ends at the first jump to a distance more than
`shell_gap` times the previous one. Distances within 10 % of each other (`shell_gap = 1.1`) therefore belong to the same
shell, e.g. the first-shell bonds of a distorted reference structure. Without a second shell, the cutoff is the
nearest-neighbor distance (as `get_nn_thresholds`). With `sepNN = false`, the cutoffs are zero, as in
`get_nn_thresholds`.

The distances are collected in bins of `bin_width` (minimum and maximum per bin) instead of being sorted, so the cost
is linear in the number of pairs. A jump by `shell_gap` cannot occur within a bin (for distances above
`bin_width / (shell_gap - 1)`), so the result is the same as with sorted distances.
"""
function get_nn_cutoffs(ions, Ts, point_grid, conf=get_empty_config(); sepNN=get_sepNN(conf), shell_gap=1.1, bin_width=0.01)
    sepNN || return get_nn_thresholds(ions, Ts, point_grid, conf, sepNN=false)
    bins = Dict{IonLabel, Dict{Int64, Tuple{Float64, Float64}}}()             # label => (bin => (min, max) distance)
    for (iion1, iion2, R) in iterate_nn_grid_points(point_grid)
        Δr = normdiff(ions[iion1].pos, ions[iion2].pos, view(Ts, :, R))
        Δr == 0 && continue
        b = floor(Int64, Δr / bin_width)
        for ion_label in (IonLabel(ions[iion1].type, ions[iion2].type, sorted=false), IonLabel(ions[iion1].type, ions[iion2].type, sorted=true))
            label_bins = get!(bins, ion_label, Dict{Int64, Tuple{Float64, Float64}}())
            lo, hi = get(label_bins, b, (Inf, -Inf))
            label_bins[b] = (min(lo, Δr), max(hi, Δr))
        end
    end
    NN_dict = Dict{IonLabel, Float64}()
    for (ion_label, label_bins) in bins
        keys_sorted = sort!(collect(keys(label_bins)))                          # at most (largest distance) / bin_width bins
        nn_distance, shell_max = label_bins[keys_sorted[1]]
        NN_dict[ion_label] = nn_distance
        for b in keys_sorted[2:end]
            lo, hi = label_bins[b]
            if lo > shell_gap * shell_max
                NN_dict[ion_label] = (shell_max + lo) / 2
                break
            end
            shell_max = hi
        end
    end
    return NN_dict
end
"""
    nn_label_statistics(strc, basis, conf=get_empty_config()) -> Dict{String, NTuple{3, Float64}}

Nearest-neighbor cutoff and number of bonds with label 1 (nearest neighbors) and 2 (farther shells) for every ion pair
(e.g. `"Si+Si"`) of a structure with `sepNN = true`, with the labels assigned as in `get_geometry_tensor` (distances
with the distortion, cutoffs from `get_nn_cutoffs`). Every bond is counted once. Ions without orbitals are skipped.
"""
function nn_label_statistics(strc, basis, conf=get_empty_config(); rcut=get_rcut(conf), rcut_tol=get_rcut_tol(conf))
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    cutoffs = get_nn_cutoffs(strc.ions, Ts, strc.point_grid, conf)
    stats = Dict{String, NTuple{3, Float64}}()
    for (iion1, iion2, R) in iterate_nn_grid_points(strc.point_grid)
        (isempty(basis.orbitals[iion1]) || isempty(basis.orbitals[iion2])) && continue
        T = view(Ts, :, R)
        r_nd = normdiff(strc.ions[iion1].pos, strc.ions[iion2].pos, T)
        r = normdiff(strc.ions[iion1].pos - strc.ions[iion1].dist, apply_pbc(strc.ions[iion2].pos, T) - strc.ions[iion2].dist)
        (r_nd == 0 || r_nd > rcut || r - abs(rcut_tol) ≥ rcut) && continue
        ion_label = IonLabel(strc.ions[iion1].type, strc.ions[iion2].type, sorted=false)
        cutoff = cutoffs[ion_label]
        nnlabel = get_nn_label(r, cutoff, conf, onsite=true, sepNN=true)
        key = string(IonLabel(strc.ions[iion1].type, strc.ions[iion2].type, sorted=true))
        _, n1, n2 = get(stats, key, (cutoff, 0.0, 0.0))
        stats[key] = (cutoff, n1 + 0.5 * (nnlabel == 1), n2 + 0.5 * (nnlabel == 2))   # every bond appears twice
    end
    return stats
end
