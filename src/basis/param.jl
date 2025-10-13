"""
    ParameterLabel(nnlabel::Int64, ion_label::IonLabel, overlap_label::SVector{3, Int64})

Represents a tight-binding (TB) parameter label by uniquely identifying it based on its nearest neighbor (NN) label, ion label, 
and overlap label.

# Fields
- `nnlabel::Int64`: An integer representing the nearest neighbor (NN) label, which identifies a specific neighbor interaction.
- `ion_label::IonLabel`: An `IonLabel` object representing the two ion types involved in the TB parameter.
- `overlap_label::SVector{3, Int64}`: A 3D static vector representing the overlap label in terms of `ll'm`.
"""
struct ParameterLabel
    nnlabel :: UInt8
    ion_label :: IonLabel
    overlap_label :: SVector{3, Int8}
end

Base.isequal(label1::ParameterLabel, label2::ParameterLabel) = label1.nnlabel == label2.nnlabel && isequal(label1.ion_label, label2.ion_label) && label1.overlap_label == label2.overlap_label

Base.hash(label::ParameterLabel, h::UInt) = hash((label.nnlabel, label.ion_label, label.overlap_label), h)

"""
    string(param_label::ParameterLabel) -> String

Returns a string representation of the `ParameterLabel` object by concatenating its nearest neighbor label (`nnlabel`), ion label (`ion_label`), and overlap label (`overlap_label`).

# Arguments
- `param_label::ParameterLabel`: The `ParameterLabel` object containing information about nearest neighbors, ions, and overlap.

# String Formatting
- Nearest neighbor (`nnlabel`) is prefixed with `"NN"` (e.g., `"NN1"`).
- Ion label is converted to string using `string(param_label.ion_label)`.
- The overlap string is constructed based on the `nnlabel` value:
    - For `nnlabel ≠ 0`: The overlap string is constructed from `ldict` and `mdict` dictionaries using the first two elements of `overlap_label`.
    - For `nnlabel == 0`:
        - If `overlap_label[3] == 0`: The overlap string is `"diag"` followed by the angular momentum index `l`.
        - Otherwise, the overlap string is `"offdiag"` followed by two angular momentum indices `l1` and `l2`.

# Returns
- A concatenated string in the form, e.g., `NN1_Ga+As_ssσ`.
"""
function Base.string(param_label::ParameterLabel)
    nn_string = "NN"*string(param_label.nnlabel)
    ion_string = string(param_label.ion_label)
    overlap_string = "_"
    if param_label.nnlabel ≠ 0
        overlap_string = ldict[param_label.overlap_label[1]]*ldict[param_label.overlap_label[2]]*mdict[param_label.overlap_label[3]]    
    elseif param_label.nnlabel == 0
        if param_label.overlap_label[3] == 0
            l = param_label.overlap_label[1]
            overlap_string = "diag"*string(l)
        else
            l1 = param_label.overlap_label[1]
            l2 = param_label.overlap_label[2]
            overlap_string = "offdiag"*string(l1)*string(l2)
        end
    end
    return nn_string*"_"*ion_string*"_"*overlap_string
end

"""
    string_to_param_label(string_label)

Convert a string parameter label `string_label` to a `ParameterLabel`.
"""
function string_to_param_label(string_label)
    nn_string, ion_string, overlap_string = split(string_label, "_")
    nnlabel = parse(Int64, nn_string[end])
    type1, type2 = string.(split(ion_string, "+"))
    ion_label = IonLabel(type1, type2)
    overlap_label = string_to_overlap_label(overlap_string)
    return ParameterLabel(nnlabel, ion_label, overlap_label)
end

"""
    same_param_label(nnlabel::Int64, ion_label::IonLabel, overlap_label::SVector{3, Int64}, param_label::ParameterLabel) -> Bool

Check if the given nearest-neighbor label (`nnlabel`), ion label (`ion_label`), and overlap label (`overlap_label`)
are the same as those in the `param_label`.

# Arguments
- `nnlabel::Int64`: The nearest-neighbor label to compare with the `nnlabel` in `param_label`.
- `ion_label::IonLabel`: The ion label to compare with the `ion_label` in `param_label`.
- `overlap_label::SVector{3, Int64}`: The overlap label to compare with the `overlap_label` in `param_label`.
- `param_label::ParameterLabel`: The parameter label to compare against.

# Returns
- `Bool`: Returns `true` if all components (`nnlabel`, `ion_label`, `overlap_label`) match those of the given `param_label`, otherwise returns `false`.
"""
function same_param_label(nnlabel, ion_label, overlap_label, param_label)
    if nnlabel == param_label.nnlabel && isequal(ion_label, param_label.ion_label) && overlap_label == param_label.overlap_label
        return true
    else
        return false
    end
end

"""
    get_ion_types_from_parameters(parameters) -> Vector{String}

Extract and return a unique list of ion types from a given list of `parameters`. 

Each parameter contains an `ion_label` that consists of two ion types concatenated by a `"+"` sign. This function splits the ion labels, and collects the unique ion types into a single vector.

# Arguments
- `parameters::Vector`: A vector of parameter objects where each parameter has an `ion_label`.

# Returns
- A vector of unique ion types (as strings) extracted from the `ion_label` field of the parameters.
"""
get_ion_types_from_parameters(parameters) = unique(vcat([split(string(parameter.ion_label), "+") for parameter in parameters]...))

"""
    get_parameters_from_overlaps(overlaps::Vector{TBOverlap}, conf::Config=get_empty_config(); sepNN::Bool=get_sepNN(conf), onsite::Bool=get_onsite(conf)) -> Vector{ParameterLabel}

Generate tight-binding parameters (`ParameterLabel`) from orbital overlaps, including onsite and nearest-neighbor interactions.

This function computes a list of `ParameterLabel` objects based on the given orbital overlaps. It determines whether to include 
onsite parameters, first-nearest-neighbor (NN) parameters, and optionally second-nearest-neighbor (NN) parameters, depending on the 
configuration.

# Arguments
- `overlaps::Vector{TBOverlap}`: A vector of orbital overlap objects representing the interactions between orbitals.
- `conf::Config`: A configuration object that specifies how parameters should be generated (default is an empty config).
- `sepNN::Bool`: A keyword argument (default from `conf`) indicating whether second-nearest-neighbor interactions should be included (`true` if included, `false` otherwise).
- `onsite::Bool`: A keyword argument (default from `conf`) indicating whether onsite parameters should be included (`true` if included, `false` otherwise).

# Returns
- A `Vector{ParameterLabel}` containing the tight-binding parameters based on the overlaps, including:
  - Onsite parameters (if `onsite=true`).
  - First-nearest-neighbor parameters (always included).
  - Second-nearest-neighbor parameters (if `sepNN=true`).
"""
function get_parameters_from_overlaps(overlaps, conf=get_empty_config(); sepNN=get_sepNN(conf), onsite=get_onsite(conf))
    parameters = ParameterLabel[]
    if onsite; get_onsite_parameters!(parameters, overlaps); end
    get_parameter_for_nn_label!(parameters, overlaps, 1)
    if sepNN; get_parameter_for_nn_label!(parameters, overlaps, 2); end
    return parameters
end

"""
    get_onsite_parameters!(parameters::Vector{ParameterLabel}, overlaps::Vector{TBOverlap})

Populate the `parameters` vector with onsite tight-binding parameters based on the given `overlaps` between orbitals.

This function identifies onsite interaction terms (where both orbitals belong to the same ion) and adds corresponding 
`ParameterLabel` objects to the `parameters` vector. These labels represent the interaction of orbitals of the same 
type (onsite) and account for angular momentum quantum numbers (`l₁`, `l₂`).

# Arguments
- `parameters::Vector{ParameterLabel}`: A vector of tight-binding parameter labels, which will be populated with new onsite parameters.
- `overlaps::Vector{TBOverlap}`: A vector of `TBOverlap` objects representing the overlap between orbitals. The function filters those with the same ion type for onsite interactions.
"""
function get_onsite_parameters!(parameters, overlaps)
    for overlap in overlaps
        if aresameions(overlap.ion_label)
            l₁, l₂ = get_baseorb_ls(overlap)
            if l₁ == l₂
                push!(parameters, ParameterLabel(0, overlap.ion_label, SVector{3, Int64}([l₁, l₂, 0])))
                if l₁ ≠ 0 && l₂ ≠ 0
                    push!(parameters, ParameterLabel(0, overlap.ion_label, SVector{3, Int64}([l₁, l₂, 1])))
                end
            else
                push!(parameters, ParameterLabel(0, overlap.ion_label, SVector{3, Int64}([l₁, l₂, 1])))
            end
        end
    end
   unique!(parameters) 
end

"""
    get_parameter_for_nn_label!(parameters::Vector{ParameterLabel}, overlaps::Vector{TBOverlap}, nn_label::Int)

Populate the `parameters` vector with tight-binding parameters for a given nearest-neighbor (NN) interaction label (`nn_label`), 
based on the provided orbital overlaps.

This function takes a list of orbital overlaps and assigns them a specified nearest-neighbor label (`nn_label`), creating 
a `ParameterLabel` for each overlap. These labels capture the interaction between orbitals at different distances (nearest-neighbor interactions).

# Arguments
- `parameters::Vector{ParameterLabel}`: A vector of tight-binding parameter labels to be populated with new nearest-neighbor parameters.
- `overlaps::Vector{TBOverlap}`: A vector of `TBOverlap` objects representing the overlaps between orbitals.
- `nn_label::Int`: The nearest-neighbor label that defines the type of interaction (e.g., first-nearest-neighbor, second-nearest-neighbor).
"""
function get_parameter_for_nn_label!(parameters, overlaps, nn_label)
    for overlap in overlaps
        overlap_label = me_to_overlap_label(overlap.type)
        push!(parameters, ParameterLabel(nn_label, overlap.ion_label, overlap_label))
    end
    unique!(parameters) 
end

"""
    write_params(parameters::Vector, parameter_values::Vector, ion_types::Vector=[], soc_parameters::Vector=[], conf::Config=get_empty_config(); filename="params")
    write_params(parameters, parameter_values, conf::Config; filename="params")

Writes the tight-binding (TB) parameters, system configuration, and optional spin-orbit coupling (SOC) parameters to a `.dat` file.

# Arguments
- `parameters::Vector`: A vector of TB parameter labels (`ParameterLabel`) that describe the system.
- `parameter_values::Vector`: A vector of parameter values corresponding to the `parameters`.
- `ion_types::Vector=[]`: A vector of ion types for which SOC parameters are defined (optional).
- `soc_parameters::Vector=[]`: A vector of SOC parameters corresponding to the ion types (optional).
- `conf::Config`: A configuration object that holds various settings for the system (e.g., system type, `rcut`, `onsite`, etc.).
- `filename::String="params"`: The base filename (without extension) for the output file (optional, default is `"params"`).

# Keyword Arguments
- `filename`: The name of the file (excluding the extension) to which the data will be written (default: `"params"`).

# File Format
The file written contains the following:
1. **System configuration block**: Includes details like `rcut`, `onsite`, `sepNN`, and alpha/n values for each unique ion type.
2. **Tight-binding parameters**: Parameters and corresponding values, formatted to align in a readable manner.
3. **SOC parameters** (optional): SOC values for each ion type, if provided.
"""
function write_params(parameters::Vector, parameter_values::Vector, ion_types::Vector=[], soc_parameters::Vector=[], conf::Config=get_empty_config(); filename="params")
    
    file = open(filename*".dat", "w")

    # Write header to file
    unique_ion_types = get_ion_types_from_parameters(parameters)
    println(file, "begin ", get_system(conf))
    println(file, "  rcut = ", get_rcut(conf))
    println(file, "  onsite = ", get_onsite(conf))
    println(file, "  sepNN = ", get_sepNN(conf))
    for ion_type in unique_ion_types
        println(file, "  $ion_type"*"_alpha = ", get_alpha(conf, ion_type))
        println(file, "  $ion_type"*"_n = ", get_n(conf, ion_type))
    end
    println(file, "end")
    println(file, "")

    # Write TB parameters to file
    for (param, param_value) in zip(parameters, parameter_values)
        L_param = length(string(param))
        println(file, string(param), " "^(25-L_param), param_value)
    end

    # Write SOC parameters to file
    for (ion_type, soc_parameter) in zip(ion_types, soc_parameters)
        L_type = length("γ_$ion_type")
        println(file, "γ_$ion_type", " "^(25-L_type), soc_parameter)
    end
    close(file)
end

write_params(parameters, parameter_values, conf::Config; filename="params") = write_params(parameters, parameter_values, [], [], conf, filename=filename)

"""
    read_params(filename="params.dat")

Reads and parses a parameter file generated by `write_params` into tight-binding (TB) parameters, parameter values, ion types, SOC parameters, and system configuration.

# Arguments
- `filename::String="params.dat"`: The name of the file to read (default is `"params.dat"`).

# Returns
- `parameters::Vector{ParameterLabel}`: A vector of TB parameter labels.
- `parameter_values::Vector{Float64}`: A vector of corresponding TB parameter values.
- `ion_types::Vector{String}`: A vector of ion types for which SOC parameters are defined (if any).
- `soc_parameters::Vector{Float64}`: A vector of SOC parameters corresponding to the `ion_types` (if any).
- `conf_values::Dict{String, String}`: A dictionary containing the configuration values from the header section (if present), such as system type, `rcut`, `onsite`, etc.
"""
function read_params(filename="params.dat")
    lines = open_and_read(filename)
    lines = split_lines(lines)
    first_param_line = 1

    # Check if there is a header
    conf_values = Dict{String, String}()
    if "begin" ∈ lines[1]
        last_header_line = next_line_with("end", lines)[1] - 1
        conf_values["system"] = lines[1][2]
        for line in lines[2:last_header_line]
            key, _, value = line
            conf_values[key] = value
        end
        first_param_line = last_header_line + 3
    end

    parameters = ParameterLabel[]
    parameter_values = Float64[]
    ion_types = String[]
    soc_parameters = Float64[]
    for line in lines[first_param_line:end]
        if !occursin("γ", line[1])
            push!(parameters, string_to_param_label(line[1]))
            push!(parameter_values, parse(Float64, line[2]))
        else
            push!(ion_types, line[1][4:end])
            push!(soc_parameters, parse(Float64, line[2]))
        end
    end
    return parameters, parameter_values, ion_types, soc_parameters, conf_values
end

"""
    check_consistency(conf_values, conf::Config) -> Bool

Checks the consistency of configuration values from a file with the given `Config` object.

# Arguments
- `conf_values::Dict{String, String}`: A dictionary of configuration values read from the file (e.g., `rcut`, `onsite`, `sepNN`, `_n`, and `_alpha` values).
- `conf::Config`: A `Config` object containing the expected values for comparison.

# Returns
- `Bool`: Returns `true` if all values in `conf_values` are consistent with the values in `conf`, otherwise returns `false`.
"""
function check_consistency(conf_values, conf::Config)
    consistent = Bool[]
    if haskey(conf_values, "rcut"); push!(consistent, conf_values["rcut"] == string(get_rcut(conf))); end
    if haskey(conf_values, "onsite"); push!(consistent, conf_values["onsite"] == string(get_onsite(conf))); end
    if haskey(conf_values, "sepNN"); push!(consistent, conf_values["sepNN"] == string(get_sepNN(conf))); end
    for (key, value) in conf_values
        if occursin("_n", key)
            ion_type = string(split(key, "_")[1])
            push!(consistent, value == string(get_n(conf, ion_type)))
        elseif occursin("_alpha", key)
            ion_type = string(split(key, "_")[1])
            push!(consistent, value == string(get_alpha(conf, ion_type)))
        end
    end
    if !all(consistent); @warn "It seems like your parameter file is not consistent with your config file. Check your results!"; end
    return all(consistent)
end