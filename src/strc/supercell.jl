"""
    get_structures(conf=get_empty_config(); index_file="config_inds.dat", xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf))

Generates a list of `Structure` objects based on the configurations from an XDATCAR file (or an h5 file) and the initial POSCAR structure. 

# Arguments
- `conf`: A configuration object, typically used to store simulation parameters. By default, it calls `get_empty_config()`.
- `index_file`: A string specifying the filename containing the configuration indices. If this file exists, the configuration indices are read from it. If not, they are sampled using `get_config_index_sample()`.
- `xdatcar`: The path to the XDATCAR file, containing configuration data for the system. 
- `sc_poscar`: The path to the POSCAR file, representing the initial supercell structure.

# Returns
- `strcs`: A vector of `Structure` objects. Each structure represents the ion positions and their displacements relative to the initial configuration.
- `config_indices`: The indices of the configurations used to create the structures.
"""
function get_structures(conf=get_empty_config(); mode="pc", index_file="config_inds.dat", xdatcar=get_xdatcar(conf), sc_poscar=get_sc_poscar(conf), poscar=get_poscar(conf))
    if lowercase(mode) == "md" || lowercase(mode) == "mixed"
        poscar = read_poscar(sc_poscar)
        lattice, configs = occursin(".h5", xdatcar) ? (h5read(xdatcar, "lattice"), h5read(xdatcar, "positions")) : read_xdatcar(xdatcar, frac=false)
        
        # Check that POSCAR lattice and XDATCAR lattice are compatible
        @assert poscar.lattice ≈ lattice

        Nconf_max = size(configs, 3)
        config_indices = index_file in readdir() ? read_from_file(index_file) : get_config_index_sample(Nconf_max, conf, val_ratio=0)[1]
        
        Ts = frac_to_cart(get_translation_vectors(1), poscar.lattice)
        rs_ion = frac_to_cart(poscar.rs_atom, poscar.lattice)
        strcs = map(config_indices) do index
            δrs_ion = similar(rs_ion)

            # Check if an atom has crossed the cell border. Distortions are otherwise not correct.
            for iion in axes(rs_ion, 2)
                Rmin = findmin([normdiff(rs_ion[:, iion], configs[:, iion, index], Ts[:, R]) for R in axes(Ts, 2)])[2]
                δrs_ion[:, iion] = rs_ion[:, iion] - configs[:, iion, index] + Ts[:, Rmin]
            end

            Structure(rs_ion, δrs_ion, poscar.atom_types, poscar.lattice, conf)
        end
        if lowercase(mode) == "md"
            return strcs, config_indices
        else
            return [Structure(conf), strcs...], config_indices
        end
    elseif lowercase(mode) == "pc"
        return [Structure(conf)], [1]
    end
end

"""
    get_config_index_sample(Nconf_max, conf=get_empty_config(); Nconf=get_Nconf(conf), Nconf_min=get_Nconf_min(conf), val_ratio=get_val_ratio(conf))

Randomly selects training and validation configuration indices from a given range of configurations.

# Arguments
- `Nconf_max`: The maximum configuration index.
- `conf`: (Optional) A configuration object from which additional parameters are obtained. Defaults to an empty configuration.
- `Nconf`: (Optional) The number of configurations to sample for training, fetched from the configuration object if not provided.
- `Nconf_min`: (Optional) The minimum configuration index, fetched from the configuration object if not provided.
- `val_ratio`: (Optional) The ratio of validation data size to training data size, fetched from the configuration object if not provided.

# Returns
- `train_config_inds`: A vector of indices for training configurations.
- `val_config_inds`: A vector of indices for validation configurations.
"""
function get_config_index_sample(Nconf_max, conf=get_empty_config(); Nconf=get_Nconf(conf), Nconf_min=get_Nconf_min(conf), val_ratio=get_val_ratio(conf))
    Nval = round(Int64, Nconf * val_ratio)
    train_config_inds = sample(Nconf_min:Nconf_max, Nconf, replace=false, ordered=true)
    remaining_indices = setdiff(Nconf_min:Nconf_max, train_config_inds)
    val_config_inds = sample(remaining_indices, Nval, replace=false, ordered=true)
    return train_config_inds, val_config_inds
end