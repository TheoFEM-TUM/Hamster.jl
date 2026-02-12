"""
    get_systems(conf) -> Vector{String}

Return the list of system names available for a given configuration `conf`.

# Arguments
- `conf`: A configuration object.

# Returns
- `Vector{String}`: A list of system names (HDF5 group names or a single system).
"""
function get_systems(conf)
    strc_file = get_xdatcar(conf)
    if isfile(strc_file) && get_train_mode(conf) == "universal"
        h5open(strc_file, "r") do file
            systems = keys(file)
            return systems
        end
    else
        return [""]
    end
end

"""
    get_structures(conf=get_empty_config(); index_file="config_inds.dat", xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf))

Generates a list of `Structure` objects based on the configurations from an XDATCAR file (or an h5 file) and the initial POSCAR structure. 

# Arguments
- `conf`: A configuration object. By default, `get_empty_config()`.
- `index_file`: A string specifying the filename containing the configuration indices. If this file exists, the configuration indices are read from it. If not, they are sampled using `get_config_index_sample()`.
- `xdatcar`: The path to the XDATCAR file, containing configuration data for the system. 
- `sc_poscar`: The path to the POSCAR file, representing the initial supercell structure.

# Returns
- `strcs`: A vector of `Structure` objects. Each structure represents the ion positions and their displacements relative to the initial configuration.
"""
function get_structures(conf=get_empty_config();
                        Rs=zeros(3, 1), 
                        mode="pc",
                        system="",
                        config_indices=[1], 
                        poscar=get_poscar(conf))
    

    if lowercase(mode) == "md" || lowercase(mode) == "mixed" || lowercase(mode) == "universal"
        rs_0, atom_types, lattice, rs_all = read_structure_file(system, conf, mode=mode)
        lattice_0 = lattice isa AbstractMatrix ? lattice : lattice[:, :, 1]
        Rs = Rs == zeros(3, 1) ? get_translation_vectors(rs_0, lattice_0, rcut=get_rcut(conf)) : Rs
        
        config_indices_ = length(config_indices) < size(rs_all, 3) ? config_indices : collect(1:size(rs_all, 3)) 
        strcs = map(config_indices_) do index
            δrs_ion = similar(rs_0)
            lattice_i = lattice isa AbstractMatrix ? lattice : lattice[:, :, index]
            Ts = frac_to_cart(get_translation_vectors(1), lattice_i)

            # Check if an atom has crossed the cell border. Distortions are otherwise not correct.
            @views for iion in axes(rs_0, 2)
                Rmin = findmin([normdiff(rs_0[:, iion], rs_all[:, iion, index], Ts[:, R]) for R in axes(Ts, 2)])[2]
                δrs_ion[:, iion] = rs_0[:, iion] - rs_all[:, iion, index] + Ts[:, Rmin]
            end

            Structure(Rs, rs_0, δrs_ion, atom_types, lattice_i, conf, system=system)
        end

        if lowercase(mode) == "md" || lowercase(mode) == "universal"
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
    read_structure_file(system, conf=get_empty_config(); mode="md", sc_poscar=get_sc_poscar(conf), xdatcar=get_xdatcar(conf))
        -> (rs_atom, atom_types, lattice, configs)

Read atomic structures and lattice information for a given system from POSCAR/XDATCAR
or HDF5 files, depending on the selected mode.

# Arguments
- `system::String`: Name of the system (material) to load, used when reading from an HDF5 file.
- `conf`: Configuration object containing simulation paths and metadata.
  Defaults to `get_empty_config()`.
- `mode::String` (keyword, default = `"md"`): Determines data source and type.
  - `"md"` → single dataset.
  - `"universal"` → read from system group in an HDF5 file.
- `sc_poscar`: Path to the POSCAR or supercell POSCAR file.
- `xdatcar`: Path to the XDATCAR or `.h5` file containing configurations.

# Returns
A tuple containing:
1. `rs_atom::Matrix{Float64}` — Cartesian positions of atoms in the reference structure.  
2. `atom_types::Vector{String}` — Atomic species labels.  
3. `lattice::Matrix{Float64}` — Lattice vectors (3×3 matrix).  
4. `configs::Array{Float64,3}` — Atomic positions for all configurations  
   (shape: `3 × N_atoms × N_configs`).
"""
function read_structure_file(system, conf=get_empty_config(); mode="md", sc_poscar=get_sc_poscar(conf), xdatcar=get_xdatcar(conf))
    if mode == "md"
        sc_poscar = read_poscar(sc_poscar)
        @unpack rs_atom, atom_types, lattice = sc_poscar
        rs_atom = frac_to_cart(rs_atom, lattice)
        if occursin(".h5", xdatcar)
            pos_key = ""
            h5open(xdatcar, "r") do file
                pos_key = haskey(file, "configs") ? "configs" : "positions"
            end
            lattice, configs = (h5read(xdatcar, "lattice"), h5read(xdatcar, pos_key))
            for n in axes(configs, 3)
                lattice_n = lattice isa Matrix ? lattice : lattice[:, :, n]
                configs[:, :, n] .= frac_to_cart(configs[:, :, n], lattice_n)
            end
        else
            lattice, configs = read_xdatcar(xdatcar, frac=false)
        end
        return rs_atom, atom_types, lattice, configs
    elseif mode == "universal"
        h5open(xdatcar, "r") do file
            system_group = file[system]
            configs = read(system_group["positions"])
            lattice = read(system_group["lattice"])
            atom_types = read(system_group["atom_types"])
            for n in axes(configs, 3)
                configs[:, :, n] .= frac_to_cart(configs[:, :, n], lattice[:, :, n])
            end
            return configs[:, :, 1], atom_types, lattice, configs
        end
    end
end

"""
    get_config_inds_for_systems(systems, comm; rank=0) -> (train_inds, val_inds)

Distribute and synchronize configuration indices for multiple systems across MPI
processes, returning training and validation index sets for each system.

# Arguments
- `systems::Vector{String}`: List of system names (e.g. materials or datasets).
- `comm`: MPI communicator used to synchronize configuration indices between ranks.
- `rank::Int` (keyword, default = `0`): Current MPI process rank.

# Returns
- `(train_config_inds, val_config_inds)`:
  Two dictionaries mapping each system name (`String`) to a vector of integer
  indices (`Vector{Int64}`) representing the selected configurations for training
  and validation, respectively.
"""
function get_config_inds_for_systems(
    systems,
    comm,
    conf=get_empty_config();
    rank=0,
    write_output=false,
    optimize=true,
)

    train_config_inds = Dict{String, Vector{Int64}}()
    val_config_inds   = Dict{String, Vector{Int64}}()

    # -------------------------------------------------
    # Open input HDF5 file ONCE (collective if using comm)
    # -------------------------------------------------
    file = nothing
    if length(systems) > 1
        file = h5open(get_xdatcar(conf), "r", comm)
    end

    for system in systems

        Nconf     = get_Nconf(conf)
        Nconf_max = get_Nconf_max(conf)

        if file !== nothing
            # All ranks execute this collectively
            Nconf_total = size(read(file[system]["positions"]), 3)

            if Nconf_total < Nconf
                Nconf = Nconf_total
            end
            if Nconf_total < Nconf_max
                Nconf_max = Nconf_total
            end
        end

        system_train_inds, system_val_inds =
            get_config_index_sample(conf; Nconf=Nconf, Nconf_max=Nconf_max)

        # -------------------------------------------------
        # Only rank 0 writes output (serial HDF5)
        # -------------------------------------------------
        if rank == 0 && write_output
            h5open("hamster_out.h5", "cw") do outfile
                g = system == "" ?
                    outfile :
                    (haskey(outfile, system) ?
                        outfile[system] :
                        create_group(outfile, system))

                if optimize
                    write(g, "train_config_inds", system_train_inds)
                    write(g, "val_config_inds", system_val_inds)
                else
                    write(g, "config_inds", system_train_inds)
                end
            end
        end

        # -------------------------------------------------
        # Broadcast to all ranks
        # -------------------------------------------------
        MPI.Bcast!(system_train_inds, comm, root=0)
        MPI.Bcast!(system_val_inds,   comm, root=0)
        MPI.Barrier(comm)

        train_config_inds[system] = system_train_inds
        val_config_inds[system]   = system_val_inds
    end

    # -------------------------------------------------
    # Close input file collectively
    # -------------------------------------------------
    if file !== nothing
        close(file)
    end

    return train_config_inds, val_config_inds
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
function get_config_index_sample(conf=get_empty_config(); 
                                Nconf=get_Nconf(conf), 
                                Nconf_min=get_Nconf_min(conf), 
                                Nconf_max=get_Nconf_max(conf),
                                validate=get_validate(conf), 
                                val_ratio=get_val_ratio(conf), 
                                train_mode=get_train_mode(conf), 
                                val_mode = get_val_mode(conf), 
                                inds_conf=get_config_inds(conf), 
                                val_inds_conf=get_val_config_inds(conf)) :: Tuple{Vector{Int64}, Vector{Int64}}

    # Training config inds
    train_config_inds = Int64[]
    if inds_conf isa Vector{Int64}
        train_config_inds = inds_conf
    elseif inds_conf isa String && occursin(".dat", inds_conf)
        train_config_inds = read_from_file(inds_conf, type=Int64)
    elseif inds_conf isa String && occursin(".h5", inds_conf)
        train_config_inds = h5read(inds_conf, "train_config_inds")
    end

    if (lowercase(train_mode) ∈ ["md", "universal"] || lowercase(train_mode) == lowercase(val_mode)) && length(train_config_inds) < Nconf
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

    if length(val_config_inds) < Nconf && (lowercase(val_mode) ∈ ["md", "universal"]) 
        Nval = train_mode == val_mode ? round(Int64, Nconf * val_ratio) : Nconf
        Nval -= length(val_config_inds)
        remaining_indices = lowercase(train_mode) == "pc" ? (Nconf_min:Nconf_max) : setdiff(Nconf_min:Nconf_max, train_config_inds)
        remaining_indices = setdiff(remaining_indices, val_config_inds)
        if length(remaining_indices) ≥ Nval
            append!(val_config_inds, sample(remaining_indices, Nval, replace=false, ordered=true))
        end
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

function split_indices_into_chunks(indices::Dict{String, Vector{T}}, nchunks; rank=0) where {T}
    all_pairs = [(sys, i) for (sys, idxs) in indices for i in idxs]

    chunk_indices = collect(chunks(all_pairs, n=nchunks))

    if length(chunk_indices) ≥ rank + 1
        local_inds = OrderedDict{String, Vector{T}}()
        for (sys, i) in chunk_indices[rank+1]
            push!(get!(local_inds, sys, T[]), i)
        end
        return local_inds
    else
        return Dict{String, Vector{T}}()
    end
end

"""
    get_number_of_bands_per_structure(bases, indices; soc=false) -> Dict{String, Int}

Compute the number of electronic bands per system, given a list of
Basis and a mapping from systems to configuration indices.

# Arguments
- `bases`: A vector of bases.
- `indices`: A `Dict{String, Vector{Int}}` containing atomic configuration indices.
- `soc` (keyword, default = `false`): If `true`, doubles the band count per to account for SOC.

# Returns
- `Dict{String, Int}`: A dictionary mapping each system name to its number of bands.
"""
function get_number_of_bands_per_structure(bases, indices; soc=false)
    Nε_all = Dict{String, Int64}()
    i = 0
    for (system, index_list) in indices
        Nε_system = Int64[]
        for index in index_list
            i += 1
            Nε = soc ? 2*length(bases[i]) : length(bases[i])
            push!(Nε_system, Nε)
        end
        @assert length(unique(Nε_system)) == 1
        Nε_all[system] = Nε_system[1]
    end
    return Nε_all
end

"""
    get_system_and_config_index(index, local_inds)

Map a global index to a specific system and configuration index.

# Arguments
- `index::Int`: The global configuration index to look up.
- `local_inds::Dict{<:Any, Vector{<:Int}}`: A dictionary mapping system names to lists of local configuration indices.

# Returns
- `(system, config_index)`: A tuple where `system` is the key corresponding to the system, and `config_index` is the local index within that system.
"""

function get_system_and_config_index(index, local_inds)
    running_ind = 0
    for (system, indices) in local_inds, config_index in indices
        running_ind += 1
        if running_ind == index
            return system, config_index
        end
    end
end