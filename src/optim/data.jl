"""
    EigData

A struct for storing eigenvalue data associated with k-points.

# Fields
- `kp::Matrix{Float64}`: A matrix where each column represents a k-point. 
- `Es::Matrix{Float64}`: A matrix where each column represents the eigenvalues corresponding to the respective k-point.
"""
struct EigData
    kp :: Matrix{Float64}
    Es :: Matrix{Float64}
end

"""
    HrData{M}

A struct for storing Hamiltonian data `Hr` and the associated lattice translation vectors `Rs` for a system, typically from Wannier90.

# Fields
- `Rs::Matrix{Float64}`: A matrix where each column represents a lattice translation vector.
- `Hr::Vector{M}`: A vector of Hamiltonian matrices `Hr`, where each entry corresponds to a lattice translation vector in `Rs`. Each element in this vector is a Hamiltonian matrix, with `M` representing the matrix type (e.g., dense or sparse).
"""
struct HrData{M}
    Rs :: Matrix{Float64}
    Hr :: Vector{M}
end

"""
    DataLoader{A, B}

A struct to store and manage training and validation datasets.

# Fields
- `train_data::Vector{A}`: A vector containing the training dataset. The type `A` represents the data type of the training set.
- `val_data::Vector{B}`: A vector containing the validation dataset. The type `B` represents the data type of the validation set.
"""
struct DataLoader
    train_data :: Vector{<:Union{EigData,HrData}}
    val_data   :: Vector{<:Union{EigData,HrData}}
end

function DataLoader(train_config_inds, val_config_inds, Nε_train, Nε_val, conf=get_empty_config(); 
                    train_path=get_train_data(conf), 
                    val_path=get_val_data(conf),
                    validate=get_validate(conf), 
                    bandmin=get_bandmin(conf),
                    bandinclude=get_bandinclude(conf),
                    bandmultiplicity = get_bandmultiplicity(conf),
                    val_bandmin=get_val_bandmin(conf), 
                    train_mode=get_train_mode(conf), 
                    val_mode=get_val_mode(conf), 
                    hr_fit=get_hr_fit(conf), 
                    eig_val=get_eig_val(conf),
                    data_unit=get_data_unit(conf))

    factor = energy_unit_factor(data_unit)
    hr_val = !eig_val
    if hr_fit
        train_data = mapreduce(vcat, train_config_inds, init=HrData[]) do (system, train_inds)
            get_hr_data(train_mode, train_path, inds=train_inds)
        end
    else
        train_data = mapreduce(vcat, train_config_inds, init=EigData[]) do (system, train_inds)
            get_eig_data(train_mode, train_path, Nε_train[system], inds=train_inds, bandmin=bandmin,
                        bandinclude=bandinclude, bandmultiplicity=bandmultiplicity, system=system)
        end
    end

    if hr_val
        val_data = mapreduce(vcat, val_config_inds, init=HrData[]) do (system, val_inds)
            get_hr_data(val_mode, val_path, inds=val_inds, empty=!validate)
        end
    else
        val_data = mapreduce(vcat, val_config_inds, init=EigData[]) do (system, val_inds)
            get_eig_data(val_mode, val_path, Nε_val[system], inds=val_inds, bandmin=val_bandmin,
            bandinclude=bandinclude, bandmultiplicity=bandmultiplicity,empty=!validate, system=system)
        end
    end
    if factor != 1
        foreach(d -> scale_energies!(d, factor), train_data)
        foreach(d -> scale_energies!(d, factor), val_data)
    end
    return DataLoader(train_data, val_data)
end

const ENERGY_UNITS_IN_EV = Dict("ev" => 1.0, "hartree" => 27.211386245988, "ry" => 13.605693122994)

"""
    energy_unit_factor(unit) -> Float64

Factor that converts energies in `unit` (`eV`, `Hartree` or `Ry`; not case sensitive) to eV.
"""
function energy_unit_factor(unit)
    haskey(ENERGY_UNITS_IN_EV, lowercase(unit)) ||
        throw(ArgumentError("unknown data_unit $unit; use one of eV, Hartree, Ry."))
    return ENERGY_UNITS_IN_EV[lowercase(unit)]
end

"""
    scale_energies!(data, factor)

Multiply the energies of `data` (the eigenvalues of `EigData`, the matrices of `HrData`) by `factor`.
"""
scale_energies!(data::EigData, factor) = (data.Es .*= factor; data)
scale_energies!(data::HrData, factor) = (foreach(H -> H .*= factor, data.Hr); data)

"""
    get_neig_and_nk(data::Vector, comm=nothing)

Get the number of eigenvalues and the number of k-points from a collection of data.

# Arguments
- `data`: A vector of either `EigData` or `HrData`.
- `comm`: (Optional) MPI communicator; if given, the data of all ranks is taken into account, so that all ranks get
  the same result.

# Returns
- `(Neig, Nk)`: The number of eigenvalues and k-points if they are the same for all data points, otherwise `(0, 0)`.
  Returns `(0, 0)` for `HrData`.
"""
function get_neig_and_nk(data::AbstractVector, comm=nothing)
    shapes = unique(map(d -> d isa EigData ? size(d.Es) : (0, 0), data))
    # (Nε, Nk) if all local data has the same shape, (0, 0) if it does not, (-1, -1) if there is no local data
    local_shape = length(shapes) == 1 ? collect(shapes[1]) : isempty(shapes) ? [-1, -1] : [0, 0]
    all_shapes = isnothing(comm) ? local_shape : MPI.Allgather(local_shape, comm)
    shapes = unique(filter(≠((-1, -1)), [(all_shapes[i], all_shapes[i+1]) for i in 1:2:length(all_shapes)]))
    return length(shapes) == 1 ? shapes[1] : (0, 0)
end

function get_band_indices(bandmin, Nε, bandinclude, bandmultiplicity; mode="pc")
    band_factor = mode == "pc" ? 1 : bandmultiplicity
    included = reduce(
        vcat,
        ((band - 1) * band_factor + 1 : band * band_factor
         for band in bandinclude);
        init=Int[],
    )

    firstband = (bandmin - 1) * band_factor + 1
    automatic = firstband : firstband + Nε - length(included) - 1

    isempty(intersect(included, automatic)) ||
        throw(ArgumentError("bandinclude overlaps with the automatic band range"))

    return sort!(vcat(included, automatic))
end

function get_eig_data(mode, path, Nε; inds=Int64[], bandmin=1, bandinclude=Int64[], bandmultiplicity=1, empty=false, system="")
    mode ∈ ("pc", "md", "universal", "mixed") || throw(ArgumentError("unknown train_mode/val_mode $mode for eigenvalue data; use pc, md, universal or mixed."))
    data = EigData[]
    band_indices = get_band_indices(bandmin, Nε, bandinclude, bandmultiplicity, mode=mode)

    if (mode == "pc" || mode == "mixed") && !empty
        push!(data, read_pc_eigenvalue_data(path, band_indices))
    end
    if (mode == "md" || mode == "mixed") && !empty
        append!(data, read_eigenvalue_data_from_path(path, inds, band_indices, Nε))
    end
    if (mode == "universal") && !empty
        append!(data, read_eigenvalue_data_from_path(path, inds, band_indices, Nε, system=system))
    end
    return data
end

function get_hr_data(mode, path; inds=Int64[], empty=false)
    mode == "universal" && !empty && throw(ArgumentError("fits and validation with Hamiltonian data (hr_fit, eig_val = false) are not supported in universal mode; use pc or md."))
    mode ∈ ("pc", "md", "universal", "mixed") || throw(ArgumentError("unknown train_mode/val_mode $mode for Hamiltonian data; use pc or md."))
    data = HrData[]
    if (mode == "pc" || mode == "mixed") && !empty
        Hr, Rs, _ = read_hrdat(path)
        push!(data, HrData(Rs, Hr))
    end
    if (mode == "md" || mode == "mixed") && !empty
        append!(data, read_hr_data_from_path(path, inds))
    end
    return data
end

function read_eigenvalue_data_from_path(path, inds, band_indices, Nε; system="")
    if occursin(".h5", path)
        h5open(path, "r") do file
            g = system == "" ? file : file[system]
            kp = read(g["kpoints"])
            Es = read(g["eigenvalues"])
            inds_ = length(inds) ≤ size(Es, 3) ? inds : collect(1:size(Es, 3))
            check_band_range(path, size(Es, 1), band_indices)
            if kp isa Matrix{Float64}
                return [EigData(kp, Es[band_indices, :, n]) for n in inds_]
            elseif kp isa Array{Float64, 3}
                return [EigData(kp[:, :, n], Es[band_indices, :, n]) for n in inds_]
            end
        end
    else
        kp, Es = read_eigenval(path)
        check_band_range(path, size(Es, 1), band_indices)
        return [EigData(kp, Es[band_indices, :])]
    end
end

"""
    read_pc_eigenvalue_data(path, band_indices) -> EigData

Read the eigenvalues of a single structure (pc mode) from a VASP `EIGENVAL` file or from an HDF5 file with the
datasets `kpoints` (3 × Nk or 3 × Nk × 1) and `eigenvalues` (Nb × Nk or Nb × Nk × 1).
"""
function read_pc_eigenvalue_data(path, band_indices)
    if occursin(".h5", path)
        kp, Es = h5read(path, "kpoints"), h5read(path, "eigenvalues")
        size(Es, 3) == 1 || throw(ArgumentError("$path contains $(size(Es, 3)) structures, but pc mode uses the data of one structure; use md mode or a file with one structure."))
        check_band_range(path, size(Es, 1), band_indices)
        return EigData(kp[:, :, 1], Es[band_indices, :, 1])
    end
    kp, Es = read_eigenval(path)
    check_band_range(path, size(Es, 1), band_indices)
    return EigData(kp, Es[band_indices, :])
end

"""
    check_band_range(path, nbands, band_indices)

Throw an `ArgumentError` if the bands selected for the fit (from `bandmin`, `bandinclude`, `bandmultiplicity` and the
number of orbitals) go beyond the `nbands` bands in the data file `path`.
"""
function check_band_range(path, nbands, band_indices)
    maximum(band_indices) ≤ nbands || throw(ArgumentError("$path has $nbands bands, but bands $(minimum(band_indices)) to $(maximum(band_indices)) are needed ($(length(band_indices)) bands, one per orbital of the structure). Check bandmin/val_bandmin, bandmultiplicity and train_mode/val_mode."))
    return nothing
end

function read_hr_data_from_path(path, inds)
    if occursin(".h5", path)
        Rs = h5read(path, "Rs")[:, :, 1]
        Hrs = h5read(path, "Hr")
        data = map(inds) do n
            @views Hr = [Hrs[:, :, R, n] for R in axes(Hrs, 3)]
            HrData(Rs, Hr)
        end
        return data
    else
        Hr, Rs, _ = read_hrdat(path)
        return [HrData(Rs, Hr)]
    end
end

"""
    get_translation_vectors_for_hr_fit(conf=get_empty_config(); hr_fit=get_hr_fit(conf), train_data=get_train_data(conf))::Matrix{Float64}

If fitting the model to Hr data, read the respective translation vectors from the `train_data` file.

# Arguments
- `conf` (default: `get_empty_config()`): A `Config` instance.
- `hr_fit`: If true, model is fit to Hr data.
- `train_data`: Path to the training data file.

# Returns
- `Rs::Matrix{Float64}`: The translation vectors, if not `hr_fit`, return zeros (Rs are calculated depending on `rcut`).
"""
function get_translation_vectors_for_hr_fit(conf=get_empty_config(); hr_fit=get_hr_fit(conf), train_data=get_train_data(conf))::Matrix{Float64}
    if hr_fit
        if occursin(".h5", train_data)
            Rs = h5read(train_data, "Rs")
            if ndims(Rs) == 3
                return Rs[:, :, 1]
            else
                return Rs
            end
        else
            _, Rs, _ = read_hrdat(train_data)
            return Rs
        end
    else
        return zeros(3, 1)
    end
end