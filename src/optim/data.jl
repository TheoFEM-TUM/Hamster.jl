struct DataLoader{A, B}
    train_data :: A
    val_data :: B
end

function DataLoader(conf=get_empty_config(); train_path=get_train_data(conf), val_path=get_val_data(conf))
    if lowercase(train_mode) == "pc"
        kp, Es = read_eigenval(train_path)
    
    elseif lowercase(train_mode) == "md"
        train_config_inds, val_config_inds = get_config_index_sample(Nconf_max, conf)
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