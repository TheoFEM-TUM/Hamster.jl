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

"""
**loss**=MAE

The `loss` tag sets the loss function to be used for the optimization.
"""
get_loss(conf::Config)::String = conf("loss", "Optimizer") == "default" ? "MAE" : conf("loss", "Optimizer")

"""
**lr**=0.1

The `lr` parameter defines the learning rate for the gradient descent parameter optimization.
"""
get_lr(conf::Config)::Float64 = conf("lr", "Optimizer") == "default" ? 0.1 : conf("lr", "Optimizer")

"""
**lr_min**=0.1

The `lr_min` parameter defines the final learning rate when applying learning rate decay.
Learning rate decay (using a cosine function) is automatically applied when `lr_min` is set to a value unequal `lr`.
"""
get_lr_min(conf::Config)::Float64 = conf("lr_min", "Optimizer") == "default" ? get_lr(conf) : conf("lr_min", "Optimizer")

"""
**niter**=1

The `niter` parameter sets the maximum number of iterations (through the training data set).
"""
get_niter(conf::Config)::Int64 = conf("niter", "Optimizer") == "default" ? 1 : conf("niter", "Optimizer")

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
    lreg=2

The `lreg` tag determines which norm is used to calculate the regularization, e.g., L2 for `lreg=2`.
"""
get_lreg(conf::Config)::Int64 = conf("lreg", "Optimizer") == "default" ? 2 : conf("lreg", "Optimizer")

"""
    lambda=0.

The `lambda` parameter determines the regularization constant.
"""
get_lambda(conf::Config)::Float64 = conf("lambda", "Optimizer") == "default" ? 0. : conf("lambda", "Optimizer")

"""
    barrier=0.

The `barrier` parameter determines at which magnitude the regularization kicks in.
"""
get_barrier(conf::Config)::Float64 = conf("barrier", "Optimizer") == "default" ? 0. : conf("barrier", "Optimizer")

"""
    train_data=EIGENVAL

The `train_data` tag sets the path (filename) that contains the training data.
"""
get_train_data(conf::Config)::String = conf("train_data", "Optimizer") == "default" ? "EIGENVAL" : conf("train_data", "Optimizer")

"""
    val_data=EIGENVAL

The `val_data` tag sets the path (filename) that contains the validation data.
"""
get_val_data(conf::Config)::String = conf("val_data", "Optimizer") == "default" ? "EIGENVAL" : conf("val_data", "Optimizer")

"""
    bandmin=1

The `bandmin` tag sets the index of the lowest band that is included in the fitting.
"""
get_bandmin(conf::Config)::Int64 = conf("bandmin", "Optimizer") == "default" ? 1 : conf("bandmin", "Optimizer")

"""
    val_bandmin=1

The `bandmin` tag sets the index of the lowest band that is included in the validation.
"""
get_val_bandmin(conf::Config)::Int64 = conf("val_bandmin", "Optimizer") == "default" ? get_bandmin(conf) : conf("val_bandmin", "Optimizer")

"""
    hr_fit=false

The `hr_fit` tag switches on fitting the effective Hamiltonian model to Hamiltonian data.
"""
get_hr_fit(conf::Config)::Bool = conf("hr_fit", "Optimizer") == "default" ? false : conf("hr_fit", "Optimizer")

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
    train_mode=PC

The `train_mode` flag switches between different optimization modes (PC, MD, mixed, multi; not case sensitive).
"""
get_train_mode(conf)::String = conf("train_mode", "Optimizer") == "default" ? "pc" : lowercase(conf("train_mode", "Optimizer"))

"""
    val_mode=PC

The `val_mode` flag switches between different modes for model validation (PC, MD, mixed, multi; not case sensitive).
"""
get_val_mode(conf::Config)::String = conf("val_mode", "Optimizer") == "default" ? "pc" : lowercase(conf("val_mode", "Optimizer"))

"""
    validate=false (true if `val_mode` is set)

The `validate` tag switches on model validation.
"""
function get_validate(conf::Config)::Bool
    if conf("validate", "Optimizer") == "default"
       return conf("val_mode", "Optimizer") == "default" && conf("val_data", "Optimizer") == "default" ? false : true
    else
        conf("validate", "Optimizer")
    end
end

"""
    val_ratio=0.2 (if validate; 0 else)

The `val_ratio` tag sets the ratio between the training set size and the validation set size.
"""
function get_val_ratio(conf::Config)::Float64
    if conf("val_ratio", "Optimizer") == "default"
        return get_validate(conf) ? 0.2 : 0.
    else
        return conf("val_ratio", "Optimizer")
    end
end

"""
    val_weights=false

If `val_weights` is true, the same weights are used for the validation loss as for the training loss.
"""
get_val_weights(conf::Config)::Bool = conf("val_weights", "Optimizer") == "default" ? false : conf("val_weights", "Optimizer")

"""
    printeachbatch=false (true if verbosity > 1)

The `printeachbatch` tag switches on printing profiler information at every batch.
"""
get_printeachbatch(conf::Config)::Bool = conf("printeachbatch", "Optimizer") == "default" ? get_verbosity(conf) > 1 : conf("printeachbatch", "Optimizer")

"""
    printeachiter=1

The `printeachiter` tag determines at what iteration interval profiler information is printed.
"""
get_printeachiter(conf::Config)::Int64 = conf("printeachiter", "Optimizer") == "default" ? 1 : conf("printeachiter", "Optimizer")

"""
    valeachiter=1

The `valeachiter` tag determines at what iteration interval a validation step is performed.
"""
get_valeachiter(conf::Config)::Int64 = conf("valeachiter", "Optimizer") == "default" ? 1 : conf("valeachiter", "Optimizer")