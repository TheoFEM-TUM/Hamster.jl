"""
    fcut(r, rcut)

Cut-off function whose value smoothly transitions to zero as `r` approaches `rcut`.
Ensures continuity by using a cosine-based smoothing function.

# Arguments
- `r`: The input distance.
- `rcut`: The cutoff radius beyond which the function returns zero.

# Returns
- A smoothly varying value between 1 and 0, with `fcut(r, rcut) = 0` for `r > rcut`.
"""
function fcut(r, rcut)
    if r > rcut
        return 0
    else
        return 1/2 * (cos(π*r/rcut) + 1)
    end
end

"""
    get_tb_descriptor(model, strc, conf)

Calculate the TB descriptor for a given a TB `model`, a structure `strc` and a TBConfig file `conf`.
"""
function get_tb_descriptor(h, V, strc::Structure, basis, conf::Config; rcut=get_ml_rcut(conf), apply_distortion=get_apply_distortion(conf), env_scale=get_env_scale(conf))
    Nε = length(basis); Norb_per_ion = size(basis); NR = size(strc.Rs, 2)

    h_env = SparseMatrixCSC{StaticVector{8, Float64}, Int64}[spzeros(StaticVector{8, Float64}, Nε, Nε) for _ in 1:NR]
    
    env = get_environmental_descriptor(h, V, strc, basis, conf)

    rs_ion = get_ion_positions(strc.ions, apply_distortion=apply_distortion)
    Ts = frac_to_cart(strc.Rs, strc.lattice)

    ij_map = get_ion_orb_to_index_map(Norb_per_ion)
    is = [Int64[] for R in 1:NR]
    js = [Int64[] for R in 1:NR]
    vals = [SVector{8, Float64}[] for R in 1:NR]
    for (iion, jion, R) in iterate_nn_grid_points(strc.point_grid)
        ri = rs_ion[iion]
        rj = rs_ion[jion] - Ts[:, R]
        Δr = normdiff(ri, rj)
        for iorb in 1:Norb_per_ion[iion], jorb in 1:Norb_per_ion[jion]
            i = ij_map[(iion, iorb)]
            j = ij_map[(jion, jorb)]
            orbswap = decide_orbswap(strc.ions[iion].type, strc.ions[jion].type, iorb, jorb)
            Zs = [element_to_number(strc.ions[iion].type), element_to_number(strc.ions[jion].type)]
            Zs = orbswap ? reverse(Zs) : Zs

            iaxis = basis.orbitals[iion][iorb].axis
            jaxis = basis.orbitals[jion][jorb].axis
            φ, θs = get_angular_descriptors(strc.ions[iion].type, strc.ions[jion].type, ri, rj, iaxis, jaxis, orbswap)
        
            if Δr ≤ rcut
                ii, jj = orbswap ? (j, i) : (i, j)
                push!(is[R], ii); push!(js[R], jj); push!(vals[R], SVector{8}(Zs[1], Zs[2], Δr, φ, θs[1], θs[2], env[ii] * env_scale, env[jj] * env_scale))
            end
        end
    end
    for R in 1:NR
        h_env[R] = sparse(is[R], js[R], vals[R])
    end

    return h_env
end

"""
    decide_orbswap(itype, jtype, iorb, jorb) -> Bool

Determines whether two orbitals should be swapped based on the types of their associated ions and their indices. The function 
enforces a consistent order of orbitals (e.g., within the descriptor vector) by swapping orbitals if they belong to the same 
ion type and `iorb > jorb`, or if the ion type of `iorb` is greater than that of `jorb`.

# Arguments
- `itype`: Type of the first ion.
- `jtype`: Type of the second ion.
- `iorb`: Index of the first orbital.
- `jorb`: Index of the second orbital.

# Returns
- `true` if the orbitals should be swapped; `false` otherwise.
"""
decide_orbswap(itype, jtype, iorb, jorb) = (itype == jtype && iorb > jorb) || (element_to_number(itype) > element_to_number(jtype))


function get_angular_descriptors(itype, jtype, ri, rj, iaxis, jaxis, orbswap)
    Δr = normdiff(ri, rj)
    Δrij = Δr > 0 ? normalize(rj - ri) : normalize(iaxis)
    Δrji = Δr > 0 ? normalize(ri - rj) : normalize(jaxis)
    φ = calc_angle(iaxis, jaxis)
    θs = Float64[calc_angle(iaxis, Δrij), calc_angle(jaxis, Δrji)]
    
    if itype == jtype 
        θs = sort(θs)
    else
        θs = orbswap ? reverse(θs) : θs
    end
    return φ, θs
end

"""
    get_environmental_descriptor(h, V, strc, basis, conf::Config; apply_params=false, rcut=get_ml_rcut(conf))

Computes the environmental descriptor for a given structure, basis, and configuration object.

# Arguments
- `h`: The geometry tensor for the given structure.
- `V`: A vector of interaction parameters.
- `strc`: The structure containing lattice and atomic positions.
- `basis`: The basis set defining the system's orbitals.
- `conf::Config`: Configuration object.

# Keyword Arguments
- `apply_params`: If `true`, uses `V` as given; otherwise, is set to ones.
- `rcut`: Cut-off radius for interactions, defaults to `get_ml_rcut(conf)`.

# Returns
- `env`: A vector representing the environmental descriptor.
"""
function get_environmental_descriptor(h, V, strc, basis, conf::Config; apply_params=false, rcut=get_ml_rcut(conf))
    Vapp = apply_params ? V : ones(length(V))
    Nε = length(basis); Norb = size(basis)

    Hr = get_hr(h, Vapp, Sparse())
    env = zeros(Nε)

    R0 = findR0(strc.Rs)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    rs_ion = get_ion_positions(strc.ions)
    index_map = get_index_to_ion_orb_map(Norb)
    @views for (R, H) in enumerate(Hr)
        for (i, j, Hval) in zip(findnz(H)...)
            iion, _ = index_map[i]
            jion, _ = index_map[j]
            Δr = normdiff(rs_ion[iion], rs_ion[jion], Ts[:, R])
            if (iion ≠ jion && R == R0) || (R ≠ R0)
                @inbounds env[i] += Hval * fcut(Δr, rcut)
            end
        end
    end
    return env
end

"""
    sample_structure_descriptors(descriptors; Ncluster=1, Npoints=1, alpha=0.5)

Selects a subset of descriptor vectors using K-Means clustering, weighted by cluster size and spread.

# Arguments
- `descriptors`: A matrix where each column represents a descriptor vector.
- `Ncluster::Int=1`: The number of clusters for K-Means.
- `Npoints::Int=1`: The total number of descriptor vectors to select.
- `alpha::Float64=0.5`: A weighting factor (0 ≤ α ≤ 1) that balances selection between cluster size (α → 1) and spread (α → 0).

# Returns
- A matrix of selected descriptor vectors with `Npoints` columns.
"""
function sample_structure_descriptors(descriptors; Ncluster=1, Npoints=1, alpha=0.5)
    result = kmeans(descriptors, Ncluster)
    indices = result.assignments
    centroids = result.centers

    cluster_sizes = [count(x -> x == c, indices) for c in 1:Ncluster]
    cluster_variances = [mean([normdiff(descriptors[:, i], centroids[:, c]) for i in findall(x -> x == c, indices)]) for c in 1:Ncluster]

    # Compute weights
    size_weights = cluster_sizes ./ sum(cluster_sizes)
    spread_weights = cluster_variances ./ sum(cluster_variances)
    final_weights = alpha .* size_weights + (1 - alpha) .* spread_weights
    final_weights ./= sum(final_weights)  # Normalize

    points_per_cluster = round.(Int, final_weights .* Npoints)
    points_per_cluster .= max.(1, points_per_cluster)

    # Adjust to ensure the exact number of `Npoints` is selected
    diff = Npoints - sum(points_per_cluster)
    if diff != 0
        sorted_clusters = sortperm(final_weights, rev=true)
        for i in 1:abs(diff)
            points_per_cluster[sorted_clusters[i]] += sign(diff)
        end
    end

    selected_indices = []
    for c in 1:Ncluster
        cluster_indices = findall(x -> x == c, indices)
        num_to_take = min(points_per_cluster[c], length(cluster_indices))
        
        # Random selection from the cluster if more points than required
        selected = sample(cluster_indices, num_to_take, replace=false)
        append!(selected_indices, selected)
    end

    return descriptors[:, selected_indices]
end