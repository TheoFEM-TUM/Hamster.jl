element_to_number(element) = elements[Symbol(element)]
number_to_element(number) = elements[number].symbol

"""
    IonLabel(types::SVector{2, Int64})

A structure representing a label for overlaps between two ions, characterized by a 2-element static vector of integers that indicate the ion species.

# Fields
- `types::SVector{2, Int64}`: A static vector containing two `Int64` values representing the types or identifiers associated with the ion. The use of `SVector{2, Int64}` ensures that the two types are stored efficiently as a fixed-size array.

# Constructors
- `IonLabel(type1::Int64, type2::Int64; sorted=true)`: Creates an `IonLabel` from two integer types. If `sorted=true`, the types will be sorted before creating the label, ensuring consistent ordering.
- `IonLabel(type1::String, type2::String; sorted=true)`: Creates an `IonLabel` from two element symbols (as `String`s). Converts the symbols to atomic numbers and optionally sorts them if `sorted=true`.
"""
struct IonLabel
    types :: SVector{2}{Int64}
end

IonLabel(type1::Int64, type2::Int64; sorted=true) = sorted ? IonLabel(sort(SVector{2, Int64}(type1, type2))) : IonLabel(SVector{2, String}(type1, type2))

IonLabel(type1::String, type2::String; sorted=true) = IonLabel(element_to_number(type1), element_to_number(type2), sorted=sorted)

"""
    Base.isequal(ion_label1::IonLabel, ion_label2::IonLabel; sorted=false) -> Bool

Compares two `IonLabel` objects for equality. 

# Arguments
- `ion_label1::IonLabel`: The first `IonLabel` instance to compare.
- `ion_label2::IonLabel`: The second `IonLabel` instance to compare.
- `sorted::Bool=false`: Whether to sort the `types` fields of both `IonLabel` instances before comparing them. Defaults to `false`. 

# Returns
- `Bool`: Returns `true` if the `IonLabel` instances are considered equal based on their `types` fields (with or without sorting), and `false` otherwise.
"""
Base.isequal(ion_label1::IonLabel, ion_label2::IonLabel; sorted=false) = sorted ? sort(ion_label1.types) == sort(ion_label2.types) : ion_label1.types == ion_label2.types

"""
    string(ion_label::IonLabel) -> String

Converts an `IonLabel` object into a human-readable string representation, where the `types` are converted into their corresponding element symbols.

# Arguments
- `ion_label::IonLabel`: The `IonLabel` instance containing two atomic types represented as atomic numbers.

# Returns
- `String`: A string representation of the `IonLabel`, where the two element numbers are converted to their respective element symbols and concatenated with a `+` sign.
"""
string(ion_label::IonLabel)::String = number_to_element(ion_label.types[1])*"+"*number_to_element(ion_label.types[2])

"""
    sameions(ion_label::IonLabel) -> Bool

Checks if both ions in the given `IonLabel` are of the same type.

# Arguments
- `ion_label::IonLabel`: The `IonLabel` object that contains two ion types represented as atomic numbers.

# Returns
- `Bool`: Returns `true` if both ions in the `IonLabel` have the same atomic number, meaning they are the same type of ion; otherwise, returns `false`.
"""
sameions(ion_label::IonLabel) = ion_label.types[1] == ion_label.types[2]

#"""
#    get_ion_label_for_matrix_element(ion_types, iion, jion, sorted)

#Return `IonLabel` for the interactions between the `iion`-th and `jion`-th ion.
#"""
#function get_ion_label_for_matrix_element(basis, i, j; sorted=true)
#    iion, _ = get_iion_and_iorb(basis, i); jion, _ = get_iion_and_iorb(basis, j)
#    return IonLabel(basis.ion_types[iion], basis.ion_types[jion], sorted=sorted)
#end

"""
    ParamLabel(nnlabel::Int64, ion_label::IonLabel, overlap_label::SVector{3, Int64})

Represents a tight-binding (TB) parameter label by uniquely identifying it based on its nearest neighbor (NN) label, ion label, 
and overlap label.

# Fields
- `nnlabel::Int64`: An integer representing the nearest neighbor (NN) label, which identifies a specific neighbor interaction.
- `ion_label::IonLabel`: An `IonLabel` object representing the two ion types involved in the TB parameter.
- `overlap_label::SVector{3, Int64}`: A 3D static vector representing the overlap label in terms of `ll'm`.
"""
struct ParamLabel
    nnlabel :: Int64
    ion_label :: IonLabel
    overlap_label :: SVector{3, Int64}
end

==(label1::ParamLabel, label2::ParamLabel) = label1.nnlabel == label2.nnlabel && isequal(label1.ion_label, label2.ion_label) && label1.overlap_label == label2.overlap_label


"""
    get_nn_label(r, r_thresh, onsite, sepNN) :: Int64

Returns the nearest neighbor (NN) label (0, 1, 2) for a matrix element based on the interatomic distance `r`, a nearest neighbor 
threshold `r_thresh`, and flags for `onsite` and `sepNN` (separated nearest neighbors).

# Arguments
- `r::Float64`: The interatomic distance between two atoms.
- `r_thresh::Float64`: The threshold distance for determining nearest neighbors.
- `onsite::Bool`: A flag indicating if onsite interactions (i.e., when `r ≈ 0`) should be considered separately.
- `sepNN::Bool`: A flag indicating if nearest neighbors should be separated into multiple groups based on distance.

# Returns
- An integer label (`Int64`) representing the type of interaction.
"""
function get_nn_label(r, r_thresh, onsite, sepNN) :: Int64
    ϵ = 1e-5
    if !onsite && !sepNN
        return 1
    elseif onsite && !sepNN
        if r ≈ 0; return 0; else return 1; end
    elseif onsite && sepNN
        if r ≈ 0; return 0
        elseif r ≤ r_thresh + ϵ; return 1
        else return 2; end
    elseif !onsite && sepNN
        if r ≤ r_thresh + ϵ; return 1
        else return 2; end
    end
end

get_nn_label(r, r_thresh, conf::TBConfig) = get_nn_label(r, r_thresh, get_onsite(conf), get_sepNN(conf))

"""
    stringtype(param_label)

Convert a `param_label` to a string.
"""
function stringtype(param_label::ParamLabel)
    nn_string = "NN"*string(param_label.nnlabel)
    ion_string = stringtype(param_label.ion_label)
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

Convert a string parameter label `string_label` to a `ParamLabel`.
"""
function string_to_param_label(string_label)
    nn_string, ion_string, overlap_string = split(string_label, "_")
    nnlabel = parse(Int64, nn_string[end])
    type1, type2 = string.(split(ion_string, "+"))
    ion_label = IonLabel(type1, type2)
    overlap_label = string_to_overlap_label(overlap_string)
    return ParamLabel(nnlabel, ion_label, overlap_label)
end

"""
    same_param_label(nnlabel, ion_label, overlap_label, param_label)

Check if `param_label` has given `nnlabel`, `ion_label` and `overlap_label`.
"""
function same_param_label(nnlabel, ion_label, overlap_label, param_label)
    if nnlabel == param_label.nnlabel && same_ion_label(ion_label, param_label.ion_label) && overlap_label == param_label.overlap_label
        return true
    else
        return false
    end
end

"""
    get_mode(type1, type2)

Return `NormalMode` if the sorted ion label is equal to the not-sorted one. Otherwise, return
`ConjugateMode`.
"""
function get_mode(type1::String, type2::String, igreaterj::Bool)
    if type1 ≠ type2
        if IonLabel(type1, type2) == IonLabel(type1, type2, sorted=false)
            return NormalMode()
        else
            return ConjugateMode()
        end
    else
        return NormalMode()
        #if igreaterj 
        #    return ConjugateMode()
        #else
        #    return NormalMode()
        #end
    end
end