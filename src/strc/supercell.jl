"""
    get_structures(conf=get_empty_config(); index_file="config_inds.dat", xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf))

Generates a list of `Structure` objects based on the configurations from an XDATCAR file (or an h5 file) and the initial POSCAR structure. 

# Arguments
- `conf`: A configuration object, typically used to store simulation parameters. By default, it calls `get_empty_config()`.
- `index_file`: A string specifying the filename containing the configuration indices. If this file exists, the configuration indices are read from it. If not, they are sampled using `get_config_index_sample()`.
- `xdatcar`: The path to the XDATCAR file, containing configuration data for the system. 
- `sc_poscar`: The path to the POSCAR file, representing the initial supercell structure.

# Returns
- `strcs`: A vector of `Structure` objects. Each structure represents the ion positions and their displacements relative to the initial configuration.
- `config_indices`: The indices of the configurations used to create the structures.
"""
function get_structures(conf=get_empty_config(); Rs=zeros(3, 1), mode="pc", config_indices=[1], xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf), poscar=get_poscar(conf))
    if lowercase(mode) == "md" || lowercase(mode) == "mixed"
        sc_poscar = read_poscar(sc_poscar)
        @unpack rs_atom, atom_types = sc_poscar

        if occursin(".h5", xdatcar)
            pos_key = ""
            h5open(xdatcar, "r") do file
                pos_key = haskey(file, "configs") ? "configs" : "positions"
            end
            lattice, configs = (h5read(xdatcar, "lattice")[:, :, 1], h5read(xdatcar, pos_key))
            for n in axes(configs, 3)
                configs[:, :, n] .= frac_to_cart(configs[:, :, n], lattice)
            end
        else
            lattice, configs = read_xdatcar(xdatcar, frac=false)
        end
        
        # Check that POSCAR lattice and XDATCAR lattice are compatible
        @assert isapprox(sc_poscar.lattice, lattice, atol=1e-3)

        Ts = frac_to_cart(get_translation_vectors(1), lattice)
        rs_ion = frac_to_cart(rs_atom, lattice)
        Rs = Rs == zeros(3, 1) ? get_translation_vectors(rs_ion, lattice, rcut=get_rcut(conf)) : Rs
        
        strcs = map(config_indices) do index
            δrs_ion = similar(rs_ion)

            # Check if an atom has crossed the cell border. Distortions are otherwise not correct.
            @views for iion in axes(rs_ion, 2)
                Rmin = findmin([normdiff(rs_ion[:, iion], configs[:, iion, index], Ts[:, R]) for R in axes(Ts, 2)])[2]
                δrs_ion[:, iion] = rs_ion[:, iion] - configs[:, iion, index] + Ts[:, Rmin]
            end

            Structure(Rs, rs_ion, δrs_ion, atom_types, lattice, conf)
        end

        if lowercase(mode) == "md"
            return strcs
        else
            return [Structure(conf), strcs...]
        end

    elseif lowercase(mode) == "pc" && length(config_indices) > 0
        pc_poscar = read_poscar(poscar)
        @unpack rs_atom, lattice = pc_poscar
        Rs = Rs == zeros(3, 1) ? get_translation_vectors(frac_to_cart(rs_atom, lattice), lattice, rcut=get_rcut(conf)) : Rs
        return [Structure(conf, Rs=Rs)]
    else
        return Structure[]
    end
end

"""
    get_config_index_sample(conf=get_empty_config(); Nconf=get_Nconf(conf), Nconf_min=get_Nconf_min(conf), Nconf_max=get_Nconf_max(conf), val_ratio=get_val_ratio(conf))

Randomly selects training and validation configuration indices from a given range of configurations.

# Arguments
- `conf`: (Optional) A configuration object from which all other parameters may be derived. Defaults to an empty configuration.
- `Nconf`: (Optional) The number of configurations to sample for training. Derived from `conf` if not specified.
- `Nconf_min`: (Optional) The minimum configuration index. Derived from `conf` if not specified.
- `Nconf_max`: (Optional) The maximum configuration index. Derived from `conf` if not specified.
- `validate`: (Optional) Boolean flag indicating whether validation should be performed. Derived from `conf` if not specified.
- `val_ratio`: (Optional) Ratio of validation configurations to training configurations. Used only if `train_mode == val_mode`.
- `train_mode`: (Optional) Mode identifier for training configurations. Derived from `conf`.
- `val_mode`: (Optional) Mode identifier for validation configurations. Derived from `conf`.

# Returns
- `train_config_inds`: A vector of indices for training configurations.
- `val_config_inds`: A vector of indices for validation configurations.
"""
function get_config_index_sample(conf=get_empty_config(); Nconf=get_Nconf(conf), Nconf_min=get_Nconf_min(conf), Nconf_max=get_Nconf_max(conf),
            validate=get_validate(conf), val_ratio=get_val_ratio(conf), train_mode=get_train_mode(conf), val_mode = get_val_mode(conf), 
            inds_conf=get_config_inds(conf), val_inds_conf=get_val_config_inds(conf))::Tuple{Vector{Int64}, Vector{Int64}}

    # Training config inds
    train_config_inds = Int64[]
    if inds_conf isa Vector{Int64}
        train_config_inds = inds_conf
    elseif inds_conf isa String && occursin(".dat", inds_conf)
        train_config_inds = read_from_file(inds_conf, type=Int64)
    elseif inds_conf isa String && occursin(".h5", inds_conf)
        train_config_inds = h5read(inds_conf, "train_config_inds")
    end

    if (lowercase(train_mode) == "md" || lowercase(train_mode) == lowercase(val_mode)) && length(train_config_inds) < Nconf
        append!(train_config_inds, sample(Nconf_min:Nconf_max, Nconf - length(train_config_inds), replace=false, ordered=true))
    elseif length(train_config_inds) == 0
        push!(train_config_inds, 1)
    end

    # Validation config inds
    val_config_inds = Int64[]
    if val_inds_conf isa Vector{Int64}
        val_config_inds = val_inds_conf
    elseif val_inds_conf isa String && occursin(".dat", val_inds_conf)
        val_config_inds = read_from_file(val_inds_conf, type=Int64)
    elseif val_inds_conf isa String && occursin(".h5", val_inds_conf)
        val_config_inds = h5read(val_inds_conf, "val_config_inds")
    end

    if length(val_config_inds) < Nconf && lowercase(val_mode) == "md"
        Nval = train_mode == val_mode ? round(Int64, Nconf * val_ratio) : Nconf
        remaining_indices = lowercase(train_mode) == "pc" ? (Nconf_min:Nconf_max) : setdiff(Nconf_min:Nconf_max, train_config_inds)
        remaining_indices = setdiff(remaining_indices, val_config_inds)
        append!(val_config_inds, sample(remaining_indices, Nval - length(val_config_inds), replace=false, ordered=true))
    end
    
    if Nconf == 1 && validate && lowercase(val_mode) == "pc"
        val_config_inds = [1] # only one config, e.g., pc
    elseif !validate
        val_config_inds = Int64[]
    end

    return train_config_inds, val_config_inds
end

"""
    split_indices_into_chunks(indices, nchunks; rank=0)

Splits a collection of indices into `nchunks` approximately equal-sized chunks and returns the chunk corresponding to the specified `rank`.

# Arguments
- `indices::AbstractVector`: The collection of indices to be split.
- `nchunks::Int`: The number of chunks to divide the indices into.
- `rank::Int=0`: The rank (0-based) specifying which chunk to return. Defaults to `0`.

# Returns
- `AbstractVector`: The chunk of indices corresponding to the specified `rank`. If the `rank` exceeds the number of chunks, an empty array of type `Int64[]` is returned.
"""
function split_indices_into_chunks(indices::AbstractVector{T}, nchunks; rank=0) where {T}
    chunk_indices = collect(chunks(indices, n=nchunks))
    if length(chunk_indices) ≥ rank + 1
       return chunk_indices[rank+1]
    else
       return T[]
    end
 end