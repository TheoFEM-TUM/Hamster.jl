const ndict = Dict(1=>R1, 2=>R2, 3=>R3, 4=>R4, 5=>R5, 6=>R6)
const mdict = Dict(0=>"σ", 1=>"π", 2=>"δ")
const mdict_inv = Dict('σ'=>0, 'π'=>1, 'δ'=>2)
const ldict = Dict(0=>"s", 1=>"p", 2=>"d")
const ldict_inv = Dict('s'=>0, 'p'=>1, 'd'=>2)

"""
    IonLabel(types::SVector{2, UInt8})

A structure representing a label for overlaps between two ions, characterized by a 2-element static vector of integers that indicate the ion species.

# Fields
- `types::SVector{2, UInt8}`: A static vector containing two `Int64` values representing the types or identifiers associated with the ion. The use of `SVector{2, Int64}` ensures that the two types are stored efficiently as a fixed-size array.

# Constructors
- `IonLabel(type1::Int64, type2::Int64; sorted=true)`: Creates an `IonLabel` from two integer types. If `sorted=true`, the types will be sorted before creating the label, ensuring consistent ordering.
- `IonLabel(type1::String, type2::String; sorted=true)`: Creates an `IonLabel` from two element symbols (as `String`s). Converts the symbols to atomic numbers and optionally sorts them if `sorted=true`.
"""
struct IonLabel
    types :: SVector{2}{UInt8}
end

IonLabel(type1::Int64, type2::Int64; sorted=true) = sorted ? IonLabel(sort(SVector{2, UInt8}(type1, type2))) : IonLabel(SVector{2, UInt8}(type1, type2))

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

Base.hash(ion_label::IonLabel, h::UInt) = hash(ion_label.types, h)

"""
    string(ion_label::IonLabel) -> String

Converts an `IonLabel` object into a human-readable string representation, where the `types` are converted into their corresponding element symbols.

# Arguments
- `ion_label::IonLabel`: The `IonLabel` instance containing two atomic types represented as atomic numbers.

# Returns
- `String`: A string representation of the `IonLabel`, where the two element numbers are converted to their respective element symbols and concatenated with a `+` sign.
"""
Base.string(ion_label::IonLabel)::String = number_to_element(ion_label.types[1])*"+"*number_to_element(ion_label.types[2])

"""
    sameions(ion_label::IonLabel) -> Bool

Checks if both ions in the given `IonLabel` are of the same type.

# Arguments
- `ion_label::IonLabel`: The `IonLabel` object that contains two ion types represented as atomic numbers.

# Returns
- `Bool`: Returns `true` if both ions in the `IonLabel` have the same atomic number, meaning they are the same type of ion; otherwise, returns `false`.
"""
aresameions(ion_label::IonLabel) = ion_label.types[1] == ion_label.types[2]

"""
    areswapped(type1, type2) -> Bool

Determines whether the two ion types `type1` and `type2` are swapped when creating an `IonLabel`. 
This function returns `true` if the unsorted `IonLabel(type1, type2, sorted=false)` differs from the sorted `IonLabel(type1, type2)`, 
indicating that the order of `type1` and `type2` was altered during sorting. Otherwise, it returns `false`.

# Arguments
- `type1`: The first ion type.
- `type2`: The second ion type.

# Returns
- `Bool`: `true` if the order of `type1` and `type2` is swapped in the sorted `IonLabel`, `false` otherwise.
"""
areswapped(type1, type2) = IonLabel(type1, type2) == IonLabel(type1, type2, sorted=false) ? false : true

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
function get_nn_label(r, r_thresh, conf=get_empty_config(); onsite=get_onsite(conf), sepNN=get_sepNN(conf)) :: Int64
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

"""
    string_to_overlap_label(overlap_string::String) -> SVector{3, Int64}

Converts an `overlap_string` into an overlap label, represented as a 3-element static vector of integers. 
The overlap label consists of two angular momentum indices (`l1`, `l2`) and a magnetic quantum number (`m`).

# Arguments
- `overlap_string::String`: A string representing the overlap label. This string can indicate diagonal or off-diagonal onsite interactions or hybridized orbitals.

# String Parsing
- If the string contains `"diag"`, it represents a diagonal or off-diagonal onsite interaction.
    - If it's diagonal, `l1` and `l2` will be the same value.
    - If it's off-diagonal, the string may represent hybrid orbitals or a combination of atomic and hybrid orbitals.
    - Based on the length of `ll_string`, `l1` and `l2` are parsed:
        - Length 2: Represents two atomic orbitals (e.g., `"diag12"`).
        - Length 3: Combination of hybrid and atomic orbitals (e.g., `"diag-12"`).
        - Length 4: Only hybrid orbitals (e.g., `"diag-1213"`).
    - `m` is set to `0` for diagonal and `1` for off-diagonal onsite interactions to differentiate between them.
- If the string does not contain `"diag"`, it represents an offsite overlap, and `l1`, `l2`, and `m` are determined using lookup tables (`ldict_inv` and `mdict_inv`).

# Returns
- `SVector{3, Int64}`: A static vector where:
    - `l1`: First angular momentum index.
    - `l2`: Second angular momentum index.
    - `m`: Magnetic quantum number (or other integer differentiating diagonal and off-diagonal cases).
"""
function string_to_overlap_label(overlap_string)
    if occursin("diag", overlap_string)
        ll_string = split(overlap_string, "g")[2]
        m = 0
        l1 = 0; l2 = 0

        # Length is one for diagonal onsites
        if !occursin("off", overlap_string)
            l1 = parse(Int64, ll_string)
            l2 = l1

        # True for offdiagonal onsite
        else
            m = 1 # diags and offdiags have different m to tell them apart

            # Length is greater two if hybrids are involved and two otherwise
            if length(ll_string) == 2
                l1 = parse(Int64, ll_string[1])
                l2 = parse(Int64, ll_string[2])

            # Length is three if hybrids are combined with atomic orbitals
            elseif length(ll_string) == 3
                if ll_string[1] == '-'
                    l1 = parse(Int64, ll_string[1:2])
                    l2 = parse(Int64, ll_string[3])
                else
                    l1 = parse(Int64, ll_string[1])
                    l2 = parse(Int64, ll_string[2:3])
                end

            # Length is four if only hybrids are involved
            elseif length(ll_string) == 4
                l1 = parse(Int64, ll_string[1:2])
                l2 = parse(Int64, ll_string[3:4])
            end
        end
        return @SVector [l1, l2, m]
    else
        l1 = ldict_inv[overlap_string[1]]
        l2 = ldict_inv[overlap_string[2]]
        m = mdict_inv[overlap_string[3]]
        return @SVector [l1, l2, m]
    end
end