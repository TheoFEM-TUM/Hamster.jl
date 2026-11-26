"""
    get_systems(conf; xdatcar=get_xdatcar(conf), mode=get_train_mode(conf)) -> Vector{String}

Return the list of system names available for a given configuration `conf`.

# Arguments
- `conf`: A configuration object.
- `xdatcar`, `mode`: the structure file and its mode (the validation trajectory `val_xdatcar` with `val_mode`).

# Returns
- `Vector{String}`: A list of system names (HDF5 group names or a single system).
"""
function get_systems(conf; xdatcar=get_xdatcar(conf), mode=get_train_mode(conf))
    strc_file = xdatcar
    if isfile(strc_file) && lowercase(mode) == "universal"
        h5open(strc_file, "r") do file
            systems = keys(file)
            return systems
        end
    else
        return [""]
    end
end

"""
    get_structures(conf=get_empty_config(); Rs=zeros(3, 1), mode="pc", system="", config_indices=[1],
                   poscar=get_poscar(conf), xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf))

Generates a list of `Structure` objects based on the configurations from an XDATCAR file (or an h5 file) and the initial POSCAR structure.

# Arguments
- `conf`: A configuration object. By default, `get_empty_config()`.
- `config_indices`: the configurations of the trajectory (md, universal and mixed mode).
- `poscar`: The path to the POSCAR of the primitive cell (pc mode).
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
                        poscar=get_poscar(conf),
                        xdatcar=get_xdatcar(conf),
                        sc_poscar=get_sc_poscar(conf))


    if lowercase(mode) == "md" || lowercase(mode) == "mixed" || lowercase(mode) == "universal"
        rs_0, atom_types, lattice, rs_all = read_structure_file(system, conf, mode=mode, xdatcar=xdatcar, sc_poscar=sc_poscar)
        lattice_0 = lattice isa AbstractMatrix ? lattice : lattice[:, :, 1]
        dmax = get_rcut_pair_selection(conf) == "frame" ? max_displacement(rs_0, rs_all, lattice) : 0.0
        if Rs == zeros(3, 1)
            Rs = get_translation_vectors(rs_0, lattice_0, rcut=rcut_search(conf, dmax))
        else
            warn_if_Rs_too_small(Rs, rs_0, lattice_0, conf, dmax)
        end

        strcs = map(config_indices) do index
            δrs_ion = similar(rs_0)
            lattice_i = lattice isa AbstractMatrix ? lattice : lattice[:, :, index]
            Ts = frac_to_cart(get_translation_vectors(1), lattice_i)

            # Check if an atom has crossed the cell border. Distortions are otherwise not correct.
            @views for iion in axes(rs_0, 2)
                Rmin = findmin([normdiff(rs_0[:, iion], rs_all[:, iion, index], Ts[:, R]) for R in axes(Ts, 2)])[2]
                δrs_ion[:, iion] = rs_0[:, iion] - apply_pbc(rs_all[:, iion, index], Ts[:, Rmin])
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
        if Rs == zeros(3, 1)
            Rs = get_translation_vectors(frac_to_cart(rs_atom, lattice), lattice, rcut=rcut_search(conf))
        else
            warn_if_Rs_too_small(Rs, frac_to_cart(rs_atom, lattice), lattice, conf)
        end
        return [Structure(conf, Rs=Rs)]
    else
        return Structure[]
    end
end

"""
    missed_bonds(strcs, conf=get_empty_config()) -> Dict{String, Float64}

Missed bonds per element pair ([`missed_bond_statistics`](@ref)), summed over the structures `strcs`. The
translation vectors of the search are built once per system.
"""
function missed_bonds(strcs, conf=get_empty_config())
    rcut, rcut_tol = get_rcut(conf), get_rcut_tol(conf)
    missed = Dict{String, Float64}()
    system_key(strc) = (strc.system, length(strc.ions))   # mixed mode: the unit cell has the system of the frames
    for skey in unique(system_key.(strcs))
        sys_strcs = filter(strc -> system_key(strc) == skey, strcs)
        dmax = maximum(norm(ion.dist) for strc in sys_strcs for ion in strc.ions; init=0.0)
        Rs_diag = get_translation_vectors(reduce(hcat, get_ion_positions(sys_strcs[1].ions)), sys_strcs[1].lattice,
                                          rcut=rcut + max(rcut_tol, 0.0) + 2 * dmax)
        for strc in sys_strcs, (key, n) in missed_bond_statistics(strc, conf; Rs_diag=Rs_diag)
            missed[key] = get(missed, key, 0.0) + n
        end
    end
    return missed
end

"""
    warn_missed_bonds(strcs, conf=get_empty_config(); comm=nothing, rank=0, label=nothing)

With `rcut_pair_selection = "reference"`, warn on `rank` 0 about missed bonds ([`missed_bonds`](@ref)) of the
structures `strcs` of all ranks of `comm` (collective). `label` names the set, e.g. `"training set"`.
"""
function warn_missed_bonds(strcs, conf=get_empty_config(); comm=nothing, rank=0, label=nothing)
    get_rcut_pair_selection(conf) == "reference" || return nothing
    missed = missed_bonds(strcs, conf)
    if !isnothing(comm)
        all_missed = MPI.gather(missed, comm, root=0)
        rank == 0 && (missed = mergewith(+, all_missed...))
    end
    rank == 0 || return nothing
    set = isnothing(label) ? "" : " of the $label"
    isempty(missed) || @warn "Missed bonds over all structures$set (within rcut, but beyond rcut in the reference): " *
        join(("$key: $(missed[key])" for key in sort(collect(keys(missed)))), ", ") *
        ". Consider rcut_pair_selection = frame."
    return nothing
end

"""
    max_displacement(rs_0, rs_all, lattice) -> Float64

Maximum over all frames `rs_all[:, :, n]` and all ions of the norm of the displacement from the reference
positions `rs_0`, taking the periodic image closest to `rs_0` (as for the distortions of the structures built in
[`get_structures`](@ref)). `lattice` is a 3x3 matrix or a 3x3xN array with the lattice vectors of each frame.
Used to size the neighbor search for `rcut_pair_selection = "frame"` (see [`rcut_search`](@ref)).
"""
function max_displacement(rs_0, rs_all, lattice)
    dmax = 0.0
    for n in axes(rs_all, 3)
        lattice_n = lattice isa AbstractMatrix ? lattice : lattice[:, :, n]
        Ts = frac_to_cart(get_translation_vectors(1), lattice_n)
        @views for iion in axes(rs_0, 2)
            Rmin = findmin([normdiff(rs_0[:, iion], rs_all[:, iion, n], Ts[:, R]) for R in axes(Ts, 2)])[2]
            dmax = max(dmax, norm(rs_0[:, iion] - apply_pbc(rs_all[:, iion, n], Ts[:, Rmin])))
        end
    end
    return dmax
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
  and validation, respectively. With `val_xdatcar`, the validation indices are those of
  [`separate_val_config_inds`](@ref), for the systems of the validation trajectory.
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

    separate = optimize && separate_val_structures(conf)
    message = rank == 0 ? structure_files_error(conf, systems, optimize=optimize) : ""
    message = MPI.bcast(message, comm, root=0)
    isempty(message) || error(message)
    file = nothing
    if length(systems) > 1
        file = h5open(get_xdatcar(conf), "r", comm)
    end

    for system in systems

        Nconf     = get_Nconf(conf)
        Nconf_max = get_Nconf_max(conf)

        if file !== nothing
            Nconf_total = size(read(file[system]["positions"]), 3)

            if Nconf_total < Nconf
                Nconf = Nconf_total
            end
            if Nconf_total < Nconf_max
                Nconf_max = Nconf_total
            end
        end

        system_train_inds, system_val_inds =
            get_config_index_sample(conf; Nconf=Nconf, Nconf_max=Nconf_max, system=system,
                                    validate=get_validate(conf) && !separate)

        MPI.Bcast!(system_train_inds, comm, root=0)
        MPI.Bcast!(system_val_inds,   comm, root=0)
        # Checked on rank 0 (it reads the frame counts) and raised on all ranks
        message = rank == 0 ? config_inds_error(system_train_inds, system_val_inds, conf, optimize=optimize, system=system,
                                                Nframes=isnothing(file) ? nothing : Nconf_total) : ""
        message = MPI.bcast(message, comm, root=0)
        isempty(message) || error(message)

        if rank == 0 && write_output
            h5open("hamster_out.h5", "cw") do outfile
                g = system == "" ?
                    outfile :
                    (haskey(outfile, system) ?
                        outfile[system] :
                        create_group(outfile, system))

                if optimize
                    write(g, "train_config_inds", system_train_inds)
                    separate || write(g, "val_config_inds", system_val_inds)
                else
                    write(g, "config_inds", system_train_inds)
                end
            end
        end

        train_config_inds[system] = system_train_inds
        separate || (val_config_inds[system] = system_val_inds)
    end

    if file !== nothing
        close(file)
    end

    separate && merge!(val_config_inds, separate_val_config_inds(conf, comm, rank=rank, write_output=write_output))

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
- `system`: (Optional) the system (universal mode), for the indices of an HDF5 file with one group per system.

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
                                val_inds_conf=get_val_config_inds(conf),
                                system="") :: Tuple{Vector{Int64}, Vector{Int64}}

    # Training config inds
    train_config_inds = read_config_inds(inds_conf, "config_inds", "train_config_inds", system=system)

    if (lowercase(train_mode) ∈ ["md", "universal"] || lowercase(train_mode) == lowercase(val_mode)) && length(train_config_inds) < Nconf
        candidates = isempty(train_config_inds) ? (Nconf_min:Nconf_max) : setdiff(Nconf_min:Nconf_max, train_config_inds)
        Nsample = Nconf - length(train_config_inds)
        Nsample ≤ length(candidates) || error("Nconf = $Nconf configurations are requested, but there are only " *
            "$(length(candidates) + length(train_config_inds)) between Nconf_min = $Nconf_min and Nconf_max = $Nconf_max " *
            "(including config_inds); increase Nconf_max (at most the number of configurations of the trajectory) or decrease Nconf.")
        append!(train_config_inds, sample(candidates, Nsample, replace=false, ordered=true))
    elseif length(train_config_inds) == 0
        push!(train_config_inds, 1)
    end

    # Validation config inds
    val_config_inds = read_config_inds(val_inds_conf, "val_config_inds", "val_config_inds", system=system)

    if length(val_config_inds) < Nconf && (lowercase(val_mode) ∈ ["md", "universal"])
        Nval = train_mode == val_mode ? round(Int64, Nconf * val_ratio) : Nconf
        Nval = max(Nval - length(val_config_inds), 0)
        remaining_indices = lowercase(train_mode) == "pc" ? (Nconf_min:Nconf_max) : setdiff(Nconf_min:Nconf_max, train_config_inds)
        remaining_indices = setdiff(remaining_indices, val_config_inds)
        if length(remaining_indices) ≥ Nval
            append!(val_config_inds, sample(remaining_indices, Nval, replace=false, ordered=true))
        elseif validate
            error("The validation set needs $Nval configurations that are not in the training set, but only " *
                  "$(length(remaining_indices)) of the configurations between Nconf_min = $Nconf_min and " *
                  "Nconf_max = $Nconf_max are left; increase Nconf_max or decrease Nconf or val_ratio.")
        end
    end
    
    if validate && lowercase(val_mode) == "pc"
        val_config_inds = [1] # the primitive cell of val_data
    elseif !validate
        val_config_inds = Int64[]
    end

    return train_config_inds, val_config_inds
end

"""
    read_config_inds(value, tag, dataset; system="") -> Vector{Int64}

The configuration indices given by the value of the tag `tag` (`config_inds` or `val_config_inds`): a list of indices,
`none` (no indices), a `.dat` file (number of indices in the first line, then one index per line) or an HDF5 file with
the dataset `dataset` (e.g. the `hamster_out.h5` of an earlier fit). For a `system` of universal mode, the dataset in
the group of the system (`<system>/<dataset>`, as in the `hamster_out.h5` of a universal fit) is read if the file has
one, else the dataset at the top of the file (the same indices for all systems).
"""
function read_config_inds(value, tag, dataset; system="")
    value isa Vector{Int64} && return copy(value)
    value == "none" && return Int64[]
    usage = "use a list of indices separated by spaces (e.g. $tag = 1 5 9), a .dat file (number of indices in the " *
            "first line, then one index per line) or an HDF5 file with the dataset $dataset"
    occursin(".dat", value) || occursin(".h5", value) || error("$tag = $value: unknown format; $usage.")
    isfile(value) || error("$tag = $value: file not found.")
    if occursin(".h5", value)
        path = h5open(value) do f
            system != "" && haskey(f, system) && f[system] isa HDF5.Group && haskey(f[system], dataset) ?
                "$system/$dataset" : haskey(f, dataset) ? dataset : ""
        end
        isempty(path) && error("$tag = $value: the file has no dataset $dataset" *
                               (system == "" ? "." : " (and no dataset $system/$dataset)."))
        return vec(h5read(value, path))
    end
    inds = try
        read_from_file(value, type=Int64)
    catch
        error("$tag = $value: the file could not be read; $usage.")
    end
    eltype(inds) == Int64 || error("$tag = $value: the file contains values that are not integers; $usage.")
    return vec(inds)
end

"""
    separate_val_config_inds(conf, comm; rank=0, write_output=false) -> Dict{String, Vector{Int64}}

Validation configurations of every system of the validation trajectory `val_xdatcar` (one system unless
`val_mode = universal`): `val_config_inds` if given (for a universal HDF5 index file, those in the group of the
system), else all configurations of the trajectory. They are independent of the training configurations. Read and
checked on rank 0 (written to `hamster_out.h5` with `write_output`) and broadcast to all ranks.
"""
function separate_val_config_inds(conf, comm; rank=0, write_output=false)
    mode, xdatcar = lowercase(get_val_mode(conf)), get_val_xdatcar(conf)
    mode ∈ ("md", "universal") || error("val_xdatcar = $xdatcar is set, but val_mode = $mode: validation structures " *
                                        "of a trajectory need val_mode = md or universal.")
    val_inds, message = Dict{String, Vector{Int64}}(), ""
    if rank == 0
        message = try
            isfile(xdatcar) || error("val_xdatcar = $xdatcar: file not found.")
            for system in get_systems(conf, xdatcar=xdatcar, mode=mode)
                Nframes = number_of_trajectory_frames(xdatcar, system)
                inds = read_config_inds(get_val_config_inds(conf), "val_config_inds", "val_config_inds", system=system)
                val_inds[system] = isempty(inds) ? collect(1:Nframes) : inds
                m = config_inds_error(Int64[], val_inds[system], conf, optimize=true, system=system, val_Nframes=Nframes)
                isempty(m) || error(m)
            end
            ""
        catch e
            e isa ErrorException ? e.msg : sprint(showerror, e)
        end
    end
    message, val_inds = MPI.bcast((message, val_inds), comm, root=0)
    isempty(message) || error(message)
    if rank == 0 && write_output
        h5open("hamster_out.h5", "cw") do outfile
            for (system, inds) in val_inds
                g = system == "" ? outfile : (haskey(outfile, system) ? outfile[system] : create_group(outfile, system))
                write(g, "val_config_inds", inds)
            end
        end
    end
    return val_inds
end

"""
    config_inds_error(train_inds, val_inds, conf; optimize, system="", Nframes=nothing, val_Nframes=nothing) -> String

Check the configuration indices of `system` (training and validation, or the structures of a standard run) against the
number of configurations of the trajectory (`xdatcar`, for the validation set `val_xdatcar` if it is set) and, in
fits, of the data files (`train_data`, `val_data`). Returns an error message, or an empty string if all indices are
valid. `Nframes` (`val_Nframes`) is the number of configurations of the trajectory (of `val_xdatcar`) if it is already
known. Indices of pc structures are not checked.
"""
function config_inds_error(train_inds, val_inds, conf; optimize, system="", Nframes=nothing, val_Nframes=nothing)
    sets = if optimize
        [("training", train_inds, lowercase(get_train_mode(conf)), get_train_data(conf), get_hr_fit(conf)),
         ("validation", val_inds, lowercase(get_val_mode(conf)), get_val_data(conf), !get_eig_val(conf))]
    else
        [("", train_inds, has_trajectory(conf) ? "md" : "pc", nothing, false)]
    end
    separate = optimize && separate_val_structures(conf)
    messages = String[]
    for (label, inds, mode, data, hr) in sets
        (mode ∈ ("md", "universal", "mixed") && !isempty(inds)) || continue
        own_file = label == "validation" && separate
        xdatcar, tag = own_file ? (get_val_xdatcar(conf), "val_xdatcar") : (get_xdatcar(conf), "xdatcar")
        isfile(xdatcar) || return "$tag = $xdatcar: file not found."
        if own_file
            val_Nframes = isnothing(val_Nframes) ? number_of_trajectory_frames(xdatcar, system) : val_Nframes
        else
            Nframes = isnothing(Nframes) ? number_of_trajectory_frames(xdatcar, system) : Nframes
        end
        Nset = own_file ? val_Nframes : Nframes
        if system == "" && !own_file && get_Nconf_max(conf) > Nframes
            push!(messages, "Nconf_max = $(get_Nconf_max(conf)), but $xdatcar has $Nframes configurations.")
        end
        outside = filter(i -> !(1 ≤ i ≤ Nset), inds)
        isempty(outside) || push!(messages, "$(lstrip("$label configurations")) $(join(outside, ", ")) are not in $xdatcar ($Nset configurations).")

        (isnothing(data) || mode == "mixed") && continue
        if !occursin(".h5", data)
            push!(messages, "$(label == "training" ? "train_data" : "val_data") = $data: $mode mode needs an HDF5 file with the data of every configuration.")
            continue
        end
        isfile(data) || continue
        Ndata = number_of_data_frames(data, system, hr=hr)
        outside = filter(i -> !(1 ≤ i ≤ Ndata), inds)
        isempty(outside) || push!(messages, "$label configurations $(join(outside, ", ")) are not in $data ($Ndata structures).")
    end
    return join(unique(messages), "\n")
end

"""
    number_of_trajectory_frames(xdatcar, system="") -> Int

Number of configurations in the trajectory file `xdatcar` (XDATCAR or HDF5 file; group `system` if not empty).
"""
function number_of_trajectory_frames(xdatcar, system="")
    occursin(".h5", xdatcar) || return count_xdatcar_configurations(xdatcar)
    h5open(xdatcar, "r") do file
        g = system == "" ? file : file[system]
        positions = g[haskey(g, "configs") ? "configs" : "positions"]
        return ndims(positions) == 3 ? size(positions, 3) : 1
    end
end

"""
    number_of_data_frames(data, system=""; hr=false) -> Int

Number of structures in the HDF5 data file `data`: eigenvalues (bands × k-points × structures), or Hamiltonians (`Hr`)
if `hr = true`.
"""
function number_of_data_frames(data, system=""; hr=false)
    h5open(data, "r") do file
        g = system == "" ? file : file[system]
        values = g[hr ? "Hr" : "eigenvalues"]
        return ndims(values) == (hr ? 4 : 3) ? size(values, ndims(values)) : 1
    end
end

"""
    h5_layout(file) -> (datasets, groups)

Names of the datasets and of the groups at the top level of the HDF5 file `file`.
"""
function h5_layout(file)
    h5open(file, "r") do f
        is_group = [f[key] isa HDF5.Group for key in keys(f)]
        return keys(f)[.!is_group], keys(f)[is_group]
    end
end

"""
    trajectory_layout_error(file, tag, mode, mode_tag) -> String

Check that the trajectory `file` (given by the input tag `tag`) has the layout that `mode` (set by `mode_tag`) reads:
in md mode an XDATCAR or an HDF5 file with the dataset `positions` (or `configs`) at the top level, in universal mode
an HDF5 file with one group per system, each with the datasets `positions`, `lattice` and `atom_types`. Returns an
error message, or an empty string (also if the file does not exist; that is reported elsewhere).
"""
function trajectory_layout_error(file, tag, mode, mode_tag)
    (mode ∈ ("md", "universal") && isfile(file)) || return ""
    universal_layout = "one group per system with the datasets positions, lattice and atom_types"
    if !occursin(".h5", file)
        mode == "md" && return ""
        return "$tag = $file: $mode_tag = universal needs an HDF5 file with $universal_layout."
    end
    datasets, groups = h5_layout(file)
    if mode == "md"
        any(∈(datasets), ("positions", "configs")) && return ""
        isempty(groups) && return "$tag = $file has no dataset positions (3 × atoms × structures) at the top level."
        return "$tag = $file has one group per system ($(join(groups, ", "))) and no dataset positions at the top " *
               "level: this is the layout of universal mode, but $mode_tag = md (a single trajectory). Use " *
               "$mode_tag = universal, or a file with the datasets positions and lattice at the top level."
    end
    if isempty(groups)
        "positions" ∈ datasets && return "$tag = $file has a single trajectory (dataset positions at the top level): " *
            "this is the layout of md mode, but $mode_tag = universal needs $universal_layout. Use $mode_tag = md, " *
            "or put the trajectory into a group named after the system."
        return "$tag = $file has no groups: $mode_tag = universal needs $universal_layout."
    end
    missing_sets = h5open(file, "r") do f
        ["$group/$name" for group in groups for name in ("positions", "lattice", "atom_types") if !haskey(f[group], name)]
    end
    isempty(missing_sets) && return ""
    return "$tag = $file: $mode_tag = universal needs $universal_layout; missing: $(join(missing_sets, ", "))."
end

"""
    data_layout_error(data, tag, mode, mode_tag, systems; hr=false) -> String

Check that the HDF5 data file `data` (input tag `tag`) has the layout that `mode` (set by `mode_tag`) reads: the
dataset `eigenvalues` (`Hr` if `hr = true`) at the top level in md mode, in a group per system of `systems` in universal
mode. Returns an error message, or an empty string (also for files that are not HDF5 files or do not exist).
"""
function data_layout_error(data, tag, mode, mode_tag, systems; hr=false)
    (mode ∈ ("md", "universal") && occursin(".h5", data) && isfile(data)) || return ""
    name = hr ? "Hr" : "eigenvalues"
    datasets, groups = h5_layout(data)
    if mode == "md"
        name ∈ datasets && return ""
        isempty(groups) && return "$tag = $data has no dataset $name at the top level."
        return "$tag = $data has one group per system ($(join(groups, ", "))) and no dataset $name at the top level: " *
               "this is the layout of universal mode, but $mode_tag = md."
    end
    missing_systems = setdiff(filter(!isempty, systems), groups)
    if !isempty(missing_systems)
        name ∈ datasets && isempty(groups) && return "$tag = $data has the dataset $name at the top level (the layout " *
            "of md mode), but $mode_tag = universal needs one group per system ($(join(missing_systems, ", ")))."
        return "$tag = $data has no group for the systems $(join(missing_systems, ", ")) of the structures."
    end
    missing_sets = h5open(data, "r") do f
        ["$system/$name" for system in filter(!isempty, systems) if !haskey(f[system], name)]
    end
    isempty(missing_sets) && return ""
    return "$tag = $data: missing datasets $(join(missing_sets, ", "))."
end

"""
    structure_files_error(conf, systems; optimize) -> String

Check that the trajectories (`xdatcar`, `val_xdatcar`) and, in fits, the HDF5 data files (`train_data`, `val_data`)
have the layout of the modes that read them (`train_mode`, `val_mode`; see [`trajectory_layout_error`](@ref) and
[`data_layout_error`](@ref)). `systems` are the systems of the training set (or of a standard run). Universal
validation structures from the trajectory of the training set need `train_mode = universal` (the systems are read
with the training mode); otherwise they need a trajectory of their own (`val_xdatcar`). Returns an error message, or an
empty string.
"""
function structure_files_error(conf, systems; optimize)
    messages = String[]
    add!(message, hint="") = isempty(message) || push!(messages, isempty(hint) ? message : "$message $hint")
    xdatcar = get_xdatcar(conf)
    if !optimize
        has_trajectory(conf) || return ""
        add!(trajectory_layout_error(xdatcar, "xdatcar", systems == [""] ? "md" : "universal", "train_mode"))
        return join(messages, "\n")
    end
    train_mode, val_mode = lowercase(get_train_mode(conf)), lowercase(get_val_mode(conf))
    add!(trajectory_layout_error(xdatcar, "xdatcar", train_mode, "train_mode"))
    add!(data_layout_error(get_train_data(conf), "train_data", train_mode, "train_mode", systems, hr=get_hr_fit(conf)))
    get_validate(conf) || return join(messages, "\n")

    val_systems = systems
    if separate_val_structures(conf)
        val_xdatcar = get_val_xdatcar(conf)
        add!(trajectory_layout_error(val_xdatcar, "val_xdatcar", val_mode, "val_mode"))
        isempty(messages) && (val_systems = get_systems(conf, xdatcar=val_xdatcar, mode=val_mode))
    elseif val_mode == "universal" && train_mode != "universal"
        add!("val_mode = universal, but train_mode = $train_mode: without val_xdatcar, the validation structures come " *
             "from the trajectory of the training set (xdatcar = $xdatcar), whose systems are only read with " *
             "train_mode = universal. Give the validation structures with val_xdatcar (Supercell block), e.g. " *
             "val_xdatcar = $xdatcar: all its configurations are then validated (or those of val_config_inds).")
        return join(messages, "\n")
    elseif val_mode != train_mode
        add!(trajectory_layout_error(xdatcar, "xdatcar", val_mode, "val_mode"),
             "(val_xdatcar is not set, so the validation structures come from xdatcar; give a trajectory of their own " *
             "with val_xdatcar in the Supercell block.)")
    end
    isempty(messages) && add!(data_layout_error(get_val_data(conf), "val_data", val_mode, "val_mode", val_systems,
                                                hr=!get_eig_val(conf)))
    return join(unique(messages), "\n")
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