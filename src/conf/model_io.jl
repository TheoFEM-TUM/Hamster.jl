"""
    save_model(ham::EffectiveHamiltonian, strcs, bases, conf::Config, comm=nothing; filename=nothing, rank=0)

Save the complete model state to an HDF5 file, including all model parameters and hyperparameters.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian containing all models.
- `strcs`: The structures.
- `bases`: The basis sets.
- `conf::Config`: The configuration object.
- `comm`: MPI communicator (optional).
- `filename`: Output filename (defaults to get_model_file(conf)).
- `rank`: MPI rank (default 0).
"""
function save_model(ham::EffectiveHamiltonian, strcs, bases, conf::Config, comm=nothing; filename=nothing, rank=0)
    isnothing(filename) && (filename = get_model_file(conf))
    
    # Only rank 0 writes the file
    if rank != 0
        return
    end
    
    # Collect all species from bases
    species = unique(vcat([get_ion_types_from_parameters(basis.parameters) for basis in bases]...))
    
    # Open HDF5 file
    if isnothing(comm)
        h5open(filename, "w") do file
            _save_model(file, ham, strcs, bases, conf, species)
        end
    else
        h5open(filename, "w", comm) do file
            _save_model(file, ham, strcs, bases, conf, species)
        end
    end
    @info "Model saved to $filename"
end

function _save_model(file, ham::EffectiveHamiltonian, strcs, bases, conf::Config, species)
    # Write version
    file["version"] = "1.0"
    
    # Write metadata
    file["metadata/system"] = conf("system")
    file["metadata/timestamp"] = Dates.now()
    
    # Write config - only model-relevant parameters
    _save_config(file, conf, species)
    
    # Write model data
    _save_model_data(file, ham, bases, conf)
end

"""
    _save_config(file, conf::Config, species)

Save model-relevant configuration parameters to the HDF5 file.
"""
function _save_config(file, conf::Config, species)
    # Options block - only model-relevant
    opts_group = create_group(file, "config/Options")
    _save_config_value(opts_group, "rcut", conf("rcut"))
    _save_config_value(opts_group, "rcut_tol", conf("rcut_tol"))
    _save_config_value(opts_group, "onsite", conf("onsite"))
    _save_config_value(opts_group, "sepNN", conf("sepNN"))
    _save_config_value(opts_group, "system", conf("system"))
    
    # Basis block
    basis_group = create_group(file, "config/Basis")
    _save_config_value(basis_group, "interpolate_rllm", conf("interpolate_rllm"))
    _save_config_value(basis_group, "itp_Ninit", conf("itp_Ninit"))
    _save_config_value(basis_group, "itp_Nmax", conf("itp_Nmax"))
    _save_config_value(basis_group, "itp_tol", conf("itp_tol"))
    _save_config_value(basis_group, "load_rllm", conf("load_rllm"))
    _save_config_value(basis_group, "rllm_file", conf("rllm_file"))
    
    # Atom block - per species
    atom_group = create_group(file, "config/atom")
    for sp in species
        sp_group = create_group(atom_group, sp)
        _save_config_value(sp_group, "orbitals", conf("orbitals", sp))
        _save_config_value(sp_group, "alpha", conf("alpha", sp))
        for (l, letter) in enumerate("spd")
            key = "alpha_$letter"
            val = conf(key, sp)
            if val != "default"
                _save_config_value(sp_group, key, val)
            end
        end
        _save_config_value(sp_group, "n", conf("n", sp))
        _save_config_value(sp_group, "nnaxes", conf("nnaxes", sp))
    end
    
    # Ewald block - if present
    if haskey(conf.blocks, "Ewald")
        ewald_group = create_group(file, "config/Ewald")
        _save_config_value(ewald_group, "method", conf("method", "Ewald"))
        _save_config_value(ewald_group, "rcut", conf("rcut", "Ewald"))
        _save_config_value(ewald_group, "alpha", conf("alpha", "Ewald"))
        _save_config_value(ewald_group, "charge_scale", conf("charge_scale", "Ewald"))
        _save_config_value(ewald_group, "mesh_spacing", conf("mesh_spacing", "Ewald"))
        _save_config_value(ewald_group, "slab_axis", conf("slab_axis", "Ewald"))
        _save_config_value(ewald_group, "charge_compensation", conf("charge_compensation", "Ewald"))
        _save_config_value(ewald_group, "centering", conf("centering", "Ewald"))
        _save_config_value(ewald_group, "site_rcut", conf("site_rcut", "Ewald"))
        _save_config_value(ewald_group, "coupling", conf("coupling", "Ewald"))
        _save_config_value(ewald_group, "update", conf("update", "Ewald"))
        _save_config_value(ewald_group, "update_polarizability", conf("update_polarizability", "Ewald"))
    end
    
    # SOC block - if present
    if haskey(conf.blocks, "SOC")
        soc_group = create_group(file, "config/SOC")
        _save_config_value(soc_group, "update", conf("update", "SOC"))
        _save_config_value(soc_group, "init_params", conf("init_params", "SOC"))
    end
    
    # ML block - if present
    if haskey(conf.blocks, "ML")
        ml_group = create_group(file, "config/ML")
        _save_config_value(ml_group, "update", conf("update", "ML"))
        _save_config_value(ml_group, "mode", conf("mode", "ML"))
        _save_config_value(ml_group, "filename", conf("filename", "ML"))
        _save_config_value(ml_group, "npoints", conf("npoints", "ML"))
        _save_config_value(ml_group, "ncluster", conf("ncluster", "ML"))
        _save_config_value(ml_group, "sim_mat", conf("sim_mat", "ML"))
    end
end

"""
    _save_config_value(group, key, value)

Helper to save a value to an HDF5 group, handling different types.
"""
function _save_config_value(group, key::String, value)
    if value isa AbstractVector
        group[string(key)] = value
    elseif value isa String
        group[string(key)] = value
    elseif value isa Bool
        group[string(key)] = value
    elseif value isa Integer
        group[string(key)] = Int64(value)
    elseif value isa AbstractFloat
        group[string(key)] = Float64(value)
    else
        group[string(key)] = string(value)
    end
end

"""
    _save_model_data(file, ham::EffectiveHamiltonian, bases, conf::Config)

Save all model data (TB, SOC, Ewald, ML parameters) to HDF5 file.
"""
function _save_model_data(file, ham::EffectiveHamiltonian, bases, conf::Config)
    models = ham.models
    
    # TBModel is always first
    if !isnothing(models) && models[1] !== nothing
        tb_model = models[1]
        _save_tb_model(file, tb_model)
    end
    
    # Save SOC model if present
    for model in models
        if model isa SOCModel
            _save_soc_model(file, model)
            break  # There's only one SOCModel
        end
    end
    
    # Save Ewald model if present
    for model in models
        if model isa EwaldOnsites
            _save_ewald_model(file, model)
            break  # There's only one EwaldOnsites
        end
    end
    
    # Save ML model if present
    for model in models
        if model isa HamiltonianKernel
            _save_ml_model(file, model)
            break  # There's only one HamiltonianKernel
        end
    end
    
    # Save rllm data from bases
    _save_rllm_data(file, bases, conf)
end

"""
    _save_tb_model(file, model::TBModel)

Save TBModel data to HDF5 file.
"""
function _save_tb_model(file, model::TBModel)
    tb_group = create_group(file, "model/tb")
    tb_group["param_labels"] = string.(model.param_labels)
    tb_group["params"] = model.params
    tb_group["update"] = model.update
    tb_group["params_per_strc"] = model.params_per_strc
end

"""
    _save_soc_model(file, model::SOCModel)

Save SOCModel data to HDF5 file.
"""
function _save_soc_model(file, model::SOCModel)
    soc_group = create_group(file, "model/soc")
    soc_group["params"] = model.params
    soc_group["param_labels"] = model.param_labels
    soc_group["types_per_strc"] = model.types_per_strc
    soc_group["Rs_info"] = model.Rs_info
    soc_group["update"] = model.update
    
    # Save SOC matrices - these are complex matrices
    matrices_group = create_group(soc_group, "matrices")
    for (key, mat) in pairs(model.matrices)
        matrices_group[string(key)] = mat
    end
end

"""
    _save_ewald_model(file, model::EwaldOnsites)

Save EwaldOnsites data to HDF5 file.
"""
function _save_ewald_model(file, model::EwaldOnsites)
    ewald_group = create_group(file, "model/ewald")
    
    # Save parameters (couplings and polarizabilities combined)
    ewald_group["params"] = model.params
    
    # Save coupling labels
    ewald_group["coupling_labels"] = model.coupling_labels
    
    # Save species
    ewald_group["species"] = model.species
    
    # Save pol labels
    ewald_group["pol_labels"] = model.pol_labels
    
    # Save potentials
    potentials_group = create_group(ewald_group, "potentials")
    for (i, pot) in enumerate(model.potentials)
        potentials_group[string(i)] = pot
    end
    
    # Save dipole potentials
    dipole_group = create_group(ewald_group, "dipole_potentials")
    for (i, dip_pot) in enumerate(model.dipole_potentials)
        dipole_group[string(i)] = dip_pot
    end
    
    # Save types per structure
    ewald_group["types_per_strc"] = model.types_per_strc
    
    # Save norb per structure
    ewald_group["norb_per_strc"] = model.norb_per_strc
    
    # Save coupling per structure
    ewald_group["coupling_per_strc"] = model.coupling_per_strc
    
    # Save Rs info
    ewald_group["Rs_info"] = model.Rs_info
    
    # Save update flags
    ewald_group["update"] = model.update
    ewald_group["update_polarizability"] = model.update_polarizability
end

"""
    _save_ml_model(file, model::HamiltonianKernel)

Save HamiltonianKernel (ML model) data to HDF5 file.
"""
function _save_ml_model(file, model::HamiltonianKernel)
    ml_group = create_group(file, "model/ml")
    
    # Save parameters
    ml_group["params"] = model.params
    
    # Save data points
    data_points_group = create_group(ml_group, "data_points")
    for (i, dp) in enumerate(model.data_points)
        data_points_group[string(i)] = dp
    end
    
    # Save sim_params (similarity parameters)
    sim_params_group = create_group(ml_group, "sim_params")
    for (key, val) in pairs(model.sim_params)
        sim_params_group[string(key)] = val
    end
    
    # Save structure descriptors
    descriptors_group = create_group(ml_group, "structure_descriptors")
    for (i, desc) in enumerate(model.structure_descriptors)
        descriptors_group[string(i)] = desc
    end
    
    # Save update flag
    ml_group["update"] = model.update
end

"""
    _save_rllm_data(file, bases, conf::Config)

Save rllm (radial function) data to HDF5 file.
"""
function _save_rllm_data(file, bases, conf::Config)
    rllm_file = get_rllm_file(conf)
    
    # Only save rllm data if it exists
    if isfile(rllm_file)
        rllm_group = create_group(file, "model/rllm")
        
        if occursin(".h5", rllm_file)
            # Read from HDF5 file
            h5open(rllm_file, "r") do src_file
                for overlap in keys(src_file)
                    # Copy the dataset to the model file
                    rllm_group[overlap] = read(src_file[overlap])
                end
            end
        else
            # Read from .dat file
            # The .dat format is: overlap_name, then xs on one line, then ys on next line
            open(rllm_file, "r") do src_file
                current_overlap = nothing
                xs = Float64[]
                ys = Float64[]
                reading_xs = false
                reading_ys = false
                
                for line in eachline(src_file)
                    line = strip(line)
                    if isempty(line)
                        continue
                    end
                    
                    if startswith(line, " ") || startswith(line, "\t")
                        # Data line
                        if reading_xs
                            xs = vcat(xs, parse.(Float64, split(line)))
                        elseif reading_ys
                            ys = vcat(ys, parse.(Float64, split(line)))
                            # Save and reset
                            rllm_group[current_overlap] = hcat(xs, ys)
                            current_overlap = nothing
                            xs = Float64[]
                            ys = Float64[]
                            reading_xs = false
                            reading_ys = false
                        end
                    else
                        # New overlap name
                        current_overlap = line
                        reading_xs = true
                        xs = Float64[]
                        ys = Float64[]
                    end
                end
                
                # Save last one if needed
                if !isempty(xs) && !isempty(ys)
                    rllm_group[current_overlap] = hcat(xs, ys)
                end
            end
        end
    end
end

"""
    load_model!(conf::Config, comm=nothing; filename=nothing, rank=0)

Load model state from HDF5 file and overwrite config tags.

# Arguments
- `conf::Config`: Configuration object to update.
- `comm`: MPI communicator (optional).
- `filename`: Input filename (defaults to get_model_file(conf)).
- `rank`: MPI rank (default 0).

# Returns
- Updated Config object.
"""
function load_model!(conf::Config, comm=nothing; filename=nothing, rank=0)
    isnothing(filename) && (filename = get_model_file(conf))
    
    if !isfile(filename)
        @warn "Model file $filename not found, skipping load."
        return conf
    end
    
    # Only rank 0 reads the file and broadcasts
    if rank == 0
        if isnothing(comm)
            h5open(filename, "r") do file
                _load_model!(conf, file)
            end
        else
            h5open(filename, "r", comm) do file
                _load_model!(conf, file)
            end
        end
        @info "Model loaded from $filename"
    end
    
    # Broadcast config to all ranks if MPI is used
    if !isnothing(comm) && rank != 0
        # Broadcast updated config from rank 0
        conf.options = MPI.Bcast(conf.options, 0, comm)
        conf.blocks = MPI.Bcast(conf.blocks, 0, comm)
    end
    
    return conf
end

"""
    _load_model!(conf::Config, file)

Internal function to load model state from an open HDF5 file.
"""
function _load_model!(conf::Config, file)
    # Load Options block
    if haskey(file, "config/Options")
        opts_group = file["config/Options"]
        for key in keys(opts_group)
            val = opts_group[key][]
            conf.options[string(key)] = string(val)
        end
    end
    
    # Load Basis block
    if haskey(file, "config/Basis")
        basis_group = file["config/Basis"]
        for key in keys(basis_group)
            val = basis_group[key][]
            conf.options[string(key)] = string(val)
        end
    end
    
    # Load atom block
    if haskey(file, "config/atom")
        atom_group = file["config/atom"]
        for sp in keys(atom_group)
            sp_group = atom_group[sp]
            if !haskey(conf.blocks, string(sp))
                conf.blocks[string(sp)] = Dict{String, String}()
            end
            for key in keys(sp_group)
                val = sp_group[key][]
                conf.blocks[string(sp)][string(key)] = string(val)
            end
        end
    end
    
    # Load Ewald block
    if haskey(file, "config/Ewald")
        ewald_group = file["config/Ewald"]
        if !haskey(conf.blocks, "Ewald")
            conf.blocks["Ewald"] = Dict{String, String}()
        end
        for key in keys(ewald_group)
            val = ewald_group[key][]
            conf.blocks["Ewald"][string(key)] = string(val)
        end
    end
    
    # Load SOC block
    if haskey(file, "config/SOC")
        soc_group = file["config/SOC"]
        if !haskey(conf.blocks, "SOC")
            conf.blocks["SOC"] = Dict{String, String}()
        end
        for key in keys(soc_group)
            val = soc_group[key][]
            conf.blocks["SOC"][string(key)] = string(val)
        end
    end
    
    # Load ML block
    if haskey(file, "config/ML")
        ml_group = file["config/ML"]
        if !haskey(conf.blocks, "ML")
            conf.blocks["ML"] = Dict{String, String}()
        end
        for key in keys(ml_group)
            val = ml_group[key][]
            conf.blocks["ML"][string(key)] = string(val)
        end
    end
end

"""
    get_model_file(conf::Config)::String

Get the model file path from config.
"""
function get_model_file(conf::Config)::String
    return conf("model_file") == "default" ? "model.h5" : conf("model_file")
end

"""
    get_load_model(conf::Config)::Bool

Check if model should be loaded from file.
"""
function get_load_model(conf::Config)::Bool
    return conf("load_model") == "default" ? false : (conf("load_model") == "true" || conf("load_model") == true)
end

"""
    get_save_model(conf::Config)::Bool

Check if model should be saved to file.
"""
function get_save_model(conf::Config)::Bool
    return conf("save_model") == "default" ? false : (conf("save_model") == "true" || conf("save_model") == true)
end
