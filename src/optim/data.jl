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
                    val_bandmin=get_val_bandmin(conf), 
                    train_mode=get_train_mode(conf), 
                    val_mode=get_val_mode(conf), 
                    hr_fit=get_hr_fit(conf), 
                    eig_val=get_eig_val(conf))

    hr_val = !eig_val
    if hr_fit
        train_data = mapreduce(vcat, train_config_inds, init=HrData[]) do (system, train_inds)
            get_hr_data(train_mode, train_path, inds=train_inds)
        end
    else
        train_data = mapreduce(vcat, train_config_inds, init=EigData[]) do (system, train_inds)
            get_eig_data(train_mode, train_path, Nε_train[system], inds=train_inds, bandmin=bandmin)
        end
    end

    if hr_val
        val_data = mapreduce(vcat, val_config_inds, init=HrData[]) do (system, val_inds)
            get_hr_data(val_mode, val_path, inds=val_inds, empty=!validate)
        end
    else
        val_data = mapreduce(vcat, val_config_inds, init=EigData[]) do (system, val_inds)
            get_eig_data(val_mode, val_path, Nε_val[system], inds=val_inds, bandmin=val_bandmin, empty=!validate)
        end
    end
    return DataLoader(train_data, val_data)
end

"""
    get_neig_and_nk(data::Vector)

Get the number of eigenvalues and the number of k-points from a collection of data.

# Arguments
- `data`: A vector of either `EigData` or `HrData`.

# Returns
- `(Neig, Nk)`: The number of eigenvalues and k-points of the first data point. `Return 0 for HrData`.
"""
function get_neig_and_nk(data::Vector{EigData})
    Nε_all = map(d->size(d.Es, 1), data)
    @show Nε_all
    if length(unique(Nε_all)) == 1
        return (size(data[1].Es, 1), size(data[1].kp, 2))
    else
        return (0, 0)
    end
end
get_neig_and_nk(data::Vector{<:HrData}) = (0, 0)

function get_eig_data(mode, path, Nε; inds=Int64[], bandmin=1, empty=false, system="")
    data = EigData[]
    if (mode == "pc" || mode == "mixed") && !empty
        kp, Es = read_eigenval(path)
        push!(data, EigData(kp, Es[bandmin:bandmin+Nε-1, :]))
    end
    if (mode == "md" || mode == "mixed") && !empty
        append!(data, read_eigenvalue_data_from_path(path, inds, bandmin, Nε))
    end
    if (mode == "universal") && !empty
        append!(data, read_eigenvalue_data_from_path(path, inds, bandmin, Nε, system=system))
    end
    return data
end

function get_hr_data(mode, path; inds=Int64[], empty=false)
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

function read_eigenvalue_data_from_path(path, inds, bandmin, Nε; system="")
    if occursin(".h5", path)
        h5open(path, "r") do file
            g = system == "" ? file : file[system]
            kp = read(g["kpoints"])
            Es = read(g["eigenvalues"])
            if kp isa Matrix{Float64}
                return [EigData(kp, Es[bandmin:bandmin+Nε-1, :, n]) for n in inds]
            elseif kp isa Array{Float64, 3}
                return [EigData(kp[:, :, n], Es[bandmin:bandmin+Nε-1, :, n]) for n in inds]
            end
        end
    else
        kp, Es = read_eigenval(path)
        return [EigData(kp, Es[bandmin:bandmin+Nε-1, :])]
    end
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