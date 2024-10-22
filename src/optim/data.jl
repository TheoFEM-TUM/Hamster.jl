struct DataLoader{A, B}
    train_data :: Vector{A}
    val_data :: Vector{B}
end

struct EigData
    kp :: Matrix{Float64}
    Es :: Matrix{Float64}
end

struct HrData{M}
    Rs :: Matrix{Float64}
    Hr :: Vector{M}
end

function DataLoader(train_config_inds, val_config_inds, PC_Nε, SC_Nε, conf=get_empty_config(); train_path=get_train_data(conf), val_path=get_val_data(conf), validate=get_validate(conf), bandmin=get_bandmin(conf), train_mode=get_train_mode(conf), val_mode=get_val_mode(conf))
    train_data = get_data(train_mode, train_path, PC_Nε, SC_Nε, inds=train_config_inds, bandmin=bandmin)
    val_data = validate ? get_data(val_mode, val_path, PC_Nε, SC_Nε, inds=val_config_inds, bandmin=bandmin) : eltype(train_data)[]
    return DataLoader(train_data, val_data)
end

function get_data(mode, path, PC_Nε, SC_Nε; inds=Int64[], bandmin=1)
    data = EigData[]
    if mode == "pc" || mode == "mixed"
        kp, Es = read_eigenval(path)
        push!(data, EigData(kp, Es[bandmin:bandmin+PC_Nε-1, :]))
    end
    if mode == "md" || mode == "mixed"
        append!(data, read_eigenvalue_data_from_path(path, inds, bandmin, SC_Nε))
    end
    return data
end

function read_eigenvalue_data_from_path(path, inds, bandmin, Nε)
    if occursin(".h5", path)
        kp = h5read(path, "kpoints")
        Es = h5read(path, "eigenvalues")
        data = [EigData(kp, Es[bandmin:bandmin+Nε-1, :, n]) for n in inds]
        return data
    else
        kp, Es = read_eigenval(path)
        return [EigData(kp, Es[bandmin:bandmin+Nε-1, :])]
    end
end