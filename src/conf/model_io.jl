"""
    save_model(ham, strcs, bases, conf, comm=nothing; filename=get_model_file(conf), rank=0)

Export fitted parameters, model settings and radial tables to a portable HDF5 file.
All ranks in `comm` must participate. Geometry-dependent tensors are rebuilt for the
structures in the receiving calculation. The file is replaced only after a successful write.
"""
function save_model(ham::EffectiveHamiltonian, strcs, bases, conf::Config, comm=nothing;
                    filename=nothing, rank=0)
    isnothing(filename) && (filename = get_model_file(conf))
    # Each rank may own radial tables absent from the other ranks.
    tables = model_radial_tables(conf, comm, rank)
    local_overlaps = any(model -> model isa TBModel, ham.models) ?
        unique([string(overlap; apply_oc=true) for basis in bases for overlap in basis.overlaps]) : String[]
    required = isnothing(comm) ? [local_overlaps] : MPI.gather(local_overlaps, comm; root=0)
    payload = nothing
    if rank == 0
        tmp = nothing
        try
            available = Set(key for records in tables for key in keys(records))
            missing = setdiff(Set(Iterators.flatten(required)), available)
            isempty(missing) || error("Missing radial tables: $(join(missing, ", "))")
            tmp, io = mktemp(dirname(abspath(filename)))
            close(io)
            h5open(tmp, "w") do file
                file["version"] = "1.0"
                file["metadata/timestamp"] = string(Dates.now())
                save_model_config(file, ham, conf)
                for model in ham.models
                    save_model_parameters(file, model)
                end
                group = create_group(file, "model/rllm")
                for records in tables, (key, data) in records
                    haskey(group, key) || (group[key] = data)
                end
            end
            mv(tmp, filename; force=true)
        catch err
            payload = sprint(showerror, err)
        finally
            isnothing(tmp) || !isfile(tmp) || rm(tmp)
        end
    end
    isnothing(comm) || (payload = MPI.bcast(payload, 0, comm))
    isnothing(payload) || error("Saving model failed: $payload")
    return nothing
end

function model_radial_tables(conf, comm, rank)
    records = Dict{String, Matrix{Float64}}()
    failure = nothing
    try
        filename = get_rllm_file(conf)
        paths = [filename]
        if !isnothing(comm) && MPI.Comm_size(comm) > 1
            push!(paths, rank_rllm_filename(filename, rank))
        end
        for path in unique(paths)
            isfile(path) || continue
            if endswith(path, ".h5") || (get_load_model(conf) && path == get_model_file(conf))
                h5open(path, "r") do file
                    group = haskey(file, "model/rllm") ? file["model/rllm"] : file
                    for key in keys(group)
                        records[key] = read(group[key])
                    end
                end
            else
                # Use the same parser as geometry construction for legacy text tables.
                splines = read_rllm(TBOverlap[]; filename=path)
                for (key, spline) in splines
                    records[key] = hcat(spline.xs, spline.ys)
                end
            end
        end
    catch err
        failure = sprint(showerror, err)
    end
    failed = isnothing(failure) ? 0 : 1
    isnothing(comm) || (failed = MPI.Allreduce(failed, max, comm))
    failed != 0 && error("Could not read model radial tables$(isnothing(failure) ? " on another rank" : ": $failure")")
    return isnothing(comm) ? [records] : MPI.gather(records, comm; root=0)
end

model_config_option(tag) = tag.block == "Options" && tag.name != "seed" &&
    (tag.category in ("basis", "model") || tag.name in ("rcut", "rcut_tol"))

function save_model_config(file, ham, conf)
    conf = deepcopy(conf)
    for model in ham.models
        if model isa Union{HamiltonianKernel,HamiltonianKernelPrecalced}
            set_value!(conf, "sim_params", "ML", model.sim_params)
        elseif model isa EwaldOnsites
            for (i, type) in enumerate(model.pol_labels)
                set_value!(conf, "polarizability", number_to_element(type), model.params[length(model.coupling_labels) + i])
            end
        end
    end
    options = create_group(file, "config/Options")
    for tag in CONFIG_TAGS
        if model_config_option(tag)
            options[lowercase(tag.name)] = get(conf.options, lowercase(tag.name), string_value(get_tag(conf, tag)))
        end
    end
    # Store atom settings verbatim, including site charges and orbital-specific couplings.
    for (block, values) in conf.blocks
        if block in ("ML", "SOC", "Ewald") || haskey(elements, Symbol(block))
            group = create_group(file, "config/blocks/$block")
            for (key, value) in values
                group[key] = value
            end
        end
    end
    file["model/tb_enabled"] = any(m -> m isa TBModel, ham.models)
    file["model/soc_enabled"] = any(m -> m isa SOCModel, ham.models)
    file["model/ewald_enabled"] = any(m -> m isa EwaldOnsites, ham.models)
    file["model/ml_enabled"] = any(m -> m isa Union{HamiltonianKernel,HamiltonianKernelPrecalced}, ham.models)
end

string_value(value::AbstractVector) = join(value, " ")
string_value(value) = string(value)

function save_model_parameters(file, model::TBModel)
    file["model/tb/labels"] = string.(model.param_labels)
    file["model/tb/params"] = model.params
end
function save_model_parameters(file, model::SOCModel)
    file["model/soc/labels"] = string.(model.param_labels)
    file["model/soc/params"] = model.params
end
function save_model_parameters(file, model::EwaldOnsites)
    file["model/ewald/labels"] = vcat(["$t,$l" for (t,l) in model.coupling_labels], ["pol,$t" for t in model.pol_labels])
    file["model/ewald/params"] = model.params
end
function save_model_parameters(file, model::Union{HamiltonianKernel,HamiltonianKernelPrecalced})
    points = model isa HamiltonianKernel ? model.data_points : model.kp.datapoints
    file["model/ml/params"] = model.params
    file["model/ml/points"] = reshape(collect(Iterators.flatten(points)), 8, length(points))
    file["model/ml/sim_params"] = model.sim_params
end

"""
    load_model!(conf, comm=nothing; filename=get_model_file(conf), rank=0)

Read and validate a model file, replacing model and atom settings in `conf` while
preserving calculation inputs and optimizer settings. Constructors then restore the
stored parameters and rebuild tensors for the requested structures. All communicator
ranks participate in the broadcast. Missing files and incompatible versions raise errors.
"""
function load_model!(conf::Config, comm=nothing; filename=nothing, rank=0)
    isnothing(filename) && (filename = get_model_file(conf))
    result = nothing
    if rank == 0
        result = try
            h5open(filename, "r") do file
                read(file["version"]) == "1.0" || error("Unsupported model file version")
                haskey(file, "model/tb_enabled") || error("Incomplete model file: missing model activation flags")
                options = Dict(key => read(file["config/Options/$key"]) for key in keys(file["config/Options"]))
                blocks = Dict{String,Dict{String,String}}()
                if haskey(file, "config/blocks")
                    for block in keys(file["config/blocks"])
                        group = file["config/blocks/$block"]
                        blocks[block] = Dict(key => read(group[key]) for key in keys(group))
                    end
                end
                flags = Dict(name => read(file["model/$(name)_enabled"]) for name in ("tb", "soc", "ewald", "ml"))
                (options, blocks, flags, nothing)
            end
        catch err
            (nothing, nothing, nothing, sprint(showerror, err))
        end
    end
    isnothing(comm) || (result = MPI.bcast(result, 0, comm))
    options, blocks, flags, failure = result
    isnothing(failure) || error("Loading model failed: $failure")
    # Mutate dictionaries because Config itself is immutable.
    for tag in CONFIG_TAGS
        model_config_option(tag) && delete!(conf.options, lowercase(tag.name))
    end
    merge!(conf.options, options)
    for block in collect(keys(conf.blocks))
        (block in ("ML", "SOC", "Ewald") || haskey(elements, Symbol(block))) && delete!(conf.blocks, block)
    end
    merge!(conf.blocks, blocks)
    for (key, value) in (("tb_model", flags["tb"]), ("soc", flags["soc"]), ("ewald", flags["ewald"]))
        set_value!(conf, key, value)
    end
    flags["ml"] && get!(conf.blocks, "ML", Dict{String,String}())
    flags["soc"] && get!(conf.blocks, "SOC", Dict{String,String}())
    flags["ewald"] && get!(conf.blocks, "Ewald", Dict{String,String}())
    if haskey(conf.blocks, "ML")
        set_value!(conf, "model", "ML", flags["ml"])
    end
    # External initialization files are superseded by the embedded weights.
    set_value!(conf, "init_params", "ones")
    for block in ("ML", "SOC", "Ewald")
        haskey(conf.blocks, block) || continue
        delete!(conf.blocks[block], "init_params")
    end
    set_value!(conf, "model_file", abspath(filename))
    set_value!(conf, "load_model", true)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", abspath(filename))
    return conf
end

# Called during construction, before descriptors and electrostatic centering use weights.
function model_parameter_values(conf, kind, labels)
    h5open(get_model_file(conf), "r") do file
        group = file["model/$kind"]
        stored_labels = read(group["labels"])
        values = read(group["params"])
        length(stored_labels) == length(values) || error("Invalid $kind parameter dimensions")
        length(unique(stored_labels)) == length(stored_labels) || error("Duplicate $kind parameter labels")
        parameters = Dict(zip(stored_labels, values))
        missing = setdiff(string.(labels), stored_labels)
        isempty(missing) || error("Model file lacks $kind parameters: $(join(missing, ", "))")
        [parameters[string(label)] for label in labels]
    end
end

function model_ml_parameters(conf)
    h5open(get_model_file(conf), "r") do file
        params = read(file["model/ml/params"])
        points = read(file["model/ml/points"])
        size(points) == (8, length(params)) || error("Invalid ML support-point dimensions")
        params, [SVector{8,Float64}(points[:,i]) for i in axes(points, 2)]
    end
end
