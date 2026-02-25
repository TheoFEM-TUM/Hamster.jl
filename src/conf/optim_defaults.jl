"""
    update_tb=haskey(conf, "Optimizer")

The `update_tb` tag determines whether/which parameters of the TB model are updated during the optimization. Defaults to `true` if an `Optimizer` block is provided.
"""
function get_update_tb(conf::Config, NV)::Vector{Bool}
    update_tb = Vector{Bool}(undef, NV)
    if conf("update_tb", "Optimizer") == "default" 
        update_tb .= haskey(conf, "Optimizer") ? true : false
    else
        if conf("update_tb", "Optimizer") isa Bool
            update_tb .= conf("update_tb", "Optimizer") ? true : false
        else
            update_tb .= map(index -> index ∈ conf("update_tb", "Optimizer"), eachindex(update_tb))
        end
    end
    return update_tb
end

@configtag loss String "MAE" " loss function for optimization (MAE, MSE)." "Optimizer"
@configtag lr Float64 0.1 "learning rate for gradient descent." "Optimizer"
@configtag lr_min Float64 get_lr(conf) "final learning rate when using learning rate decay if lr≠lr_min." "Optimizer"
@configtag niter Int64 1 "number of gradient descent steps." "Optimizer"
@configtag lreg Int64 2 "norm for regularization (1=L1, 2=L2)." "Optimizer"
@configtag lambda Float64 0. "regularization constant" "Optimizer"
@configtag barrier Float64 0. "parameter magnitude where regularization kicks in." "Optimizer"

"""
**wE**=ones

The `wE` tag sets the weight of each energy band for the calculation of the loss. Individual weights can also be set with, e.g., `wE_3 = 2`.
"""
get_band_weights(conf::Config, Nε)::Vector{Float64} = conf("wE", "Optimizer") == "default" ? ones(Nε) : conf("wE", "Optimizer")

"""
    wk=ones

The `wk` tag sets the weight of each kpoint for the calculation of the loss. Individual weights can also be set with, e.g., `wk_3 = 2`.
"""
get_kpoint_weights(conf::Config, Nk)::Vector{Float64} = conf("wk", "Optimizer") == "default" ? ones(Nk) : conf("wk", "Optimizer")

"""
*train_data* = EIGENVAL

Specify the path to the training data file.

This tag tells Hamster where to load the training dataset used during model evaluation and optimization.

# Accepted Formats

- **VASP `EIGENVAL` file**  
  Supported only in *pc* training mode. The file is read directly and converted into the internal eigenvalue representation.

- **HDF5 (`.h5`) file**  
  Must contain the datasets:
  - `eigenvalues` with shape **[Nbands × Nkpoints × Nstructures]**
  - `kpoints` with shape **[3 × Nkpoints × Nstructures]**

The datasets must correspond structure-by-structure to the training set.  
The number of bands in the data does *not* need to match the Hamster basis size; only the subset starting at `bandmin` is used.
"""
get_train_data(conf::Config)::String = conf("train_data", "Optimizer") == "default" ? "EIGENVAL" : conf("train_data", "Optimizer")
push!(CONFIG_TAGS, ConfigTag{String}("train_data", "Optimizer", conf->"EIGENVAL", "path to training data"))

"""
*val_data*=EIGENVAL

Specify the path to the validation data file.

This tag tells Hamster where to load the validation dataset used during model evaluation and optimization.

# Accepted Formats

- **VASP `EIGENVAL` file**  
  Supported only in *pc* validation mode. The file is read directly and converted into the internal eigenvalue representation.

- **HDF5 (`.h5`) file**  
  Must contain the datasets:
  - `eigenvalues` with shape **[Nbands × Nkpoints × Nstructures]**
  - `kpoints` with shape **[3 × Nkpoints × Nstructures]**

The datasets must correspond structure-by-structure to the validation set.  
The number of bands in the data does *not* need to match the Hamster basis size; only the subset starting at `val_bandmin` is used.
"""
get_val_data(conf::Config)::String = conf("val_data", "Optimizer") == "default" ? "EIGENVAL" : conf("val_data", "Optimizer")
push!(CONFIG_TAGS, ConfigTag{String}("val_data", "Optimizer", conf->"EIGENVAL", "path to validation data"))


@configtag bandmin Int64 1 "lowest band index to include in optimization." "Optimizer"
@configtag val_bandmin Int64 get_bandmin(conf) "lowest band index to include in validation set." "Optimizer"
@configtag hr_fit Bool false "switches to fitting the model to Hamiltonian data." "Optimizer"

"""
    eig_fit=true (false if hr_fit)

The `eig_fit` tag switches on fitting the effective Hamiltonian to eigenvalue data. If both `hr_fit` and `eig_fit` are true, the model is first fit to Hamiltonian data and then to eigenvalue data.
"""
function get_eig_fit(conf::Config)::Bool
    if conf("eig_fit", "Optimizer") == "default"
        return get_hr_fit(conf) ? false : true
    else
        return conf("eig_fit", "Optimizer")
    end
end

"""
    eig_val=eig_fit

The `eig_val` tag switches on validating the effective Hamiltonian model with eigenvalue data.
"""
get_eig_val(conf::Config)::Bool = conf("eig_val", "Optimizer") == "default" ? get_eig_fit(conf) : conf("eig_val", "Optimizer")

"""
**train_mode**=PC

The `train_mode` flag switches between different optimization modes (`PC`, `MD`, `universal`; not case sensitive).

Defaults:
 - `"pc"` — if no `Supercell` block is present.
 - `"md"` — if a `Supercell` block is present but only one system is found.
"""
function get_train_mode(conf)::String
    if conf("train_mode", "Optimizer") == "default" 
        if haskey(conf, "Supercell")
            return "md"
        else
            return "pc"
        end
    else
        return lowercase(conf("train_mode", "Optimizer"))
    end
end
push!(CONFIG_TAGS, ConfigTag{String}("train_mode", "Optimizer", conf->get_train_mode(conf), "training mode"))

"""
    val_mode=PC

The `val_mode` flag switches between different modes for model validation (PC, MD, mixed, multi; not case sensitive).
"""
get_val_mode(conf::Config)::String = conf("val_mode", "Optimizer") == "default" ? get_train_mode(conf) : lowercase(conf("val_mode", "Optimizer"))
push!(CONFIG_TAGS, ConfigTag{String}("val_mode", "Optimzer", conf->get_val_mode(conf), "validation mode"))

"""
    validate=false (true if `val_mode` is set)

The `validate` tag switches on model validation (usually no need to set this manually).
"""
function get_validate(conf::Config)::Bool
    if conf("validate", "Optimizer") == "default"
       return conf("val_mode", "Optimizer") == "default" && conf("val_data", "Optimizer") == "default" ? false : true
    else
        conf("validate", "Optimizer")
    end
end
push!(CONFIG_TAGS, ConfigTag{Bool}("validate", "Optimizer", conf->get_validate(conf), "switches on model validation (automatic)"))


@configtag val_ratio Float64 ifelse(get_validate(conf),0.2,0.) "ratio between training and validation set size." "Optimizer" 
@configtag val_weights Bool false "if true, same weights are used for training and validation." "Optimizer"
@configtag printeachbatch Bool get_verbosity(conf)>1 "profiler prints after each batch." "Optimizer"
@configtag printeachiter Int64 1 "iteration interval for profiler prints." "Optimizer"
@configtag valeachiter Int64 1 "iteration interval for validation." "Optimizer"