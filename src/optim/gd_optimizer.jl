"""
    GDOptimizer

A gradient descent optimizer for training Hamiltonian models, incorporating loss functions, regularization, and an Adam optimizer for parameter updates.

# Fields
- `loss::Loss`: The loss function used for training. Determines how the model's error is measured and minimized during training.
- `val_loss::Loss`: The loss function used for validation. Evaluates the model's performance on unseen data.
- `reg::Regularization`: Regularization strategy applied during training to prevent overfitting.
- `adam::Adam`: The Adam optimizer instance used for updating model parameters with gradient-based methods.
- `Niter::Int64`: The number of training iterations.

# Usage
This structure encapsulates all components required for optimizing a Hamiltonian model. It is typically passed to training functions, such as `train_step!` or `optimize_model!`, to guide the optimization process.
"""
struct GDOptimizer
    loss :: Loss
    val_loss :: Loss
    reg :: Regularization
    adam :: Adam
    Niter :: Int64
end

"""
    GDOptimizer(Nε, Nk, conf=get_empty_config(); lr=get_lr(conf), Niter=get_niter(conf))

Creates a gradient descent optimizer for a given loss function, regularization term, and optimizer configuration.

# Arguments
- `Nε::Int` (optional): Parameter representing the number of energy levels. If `Nε == 0`, a default configuration is used.
- `Nk::Int` (optional): Parameter representing the number of k-points. If `Nk == 0`, a default configuration is used.
- `conf::Any`: Configuration object for the optimizer, defaults to `get_empty_config()`.
- `lr::Float64` (optional): Learning rate for the Adam optimizer. Defaults to the value returned by `get_lr(conf)`.
- `Niter::Int` (optional): Number of iterations for the optimization process. Defaults to the value returned by `get_niter(conf)`.
- `val_weights::Bool` (optional): If true, same weights are used for validation as for training.
"""
function GDOptimizer(Nε, Nk, conf=get_empty_config(); lr=get_lr(conf), Niter=get_niter(conf), val_weights=get_val_weights(conf))
    loss = Nε == 0 && Nk == 0 ? Loss(conf) : Loss(Nε, Nk, conf)
    if val_weights
        val_loss = Nε == 0 && Nk == 0 ? Loss(conf) : Loss(Nε, Nk, conf)
    else
        val_loss = Loss(conf)
    end
    reg = Regularization(conf)
    adam = Adam(lr)
    return GDOptimizer(loss, val_loss, reg, adam, Niter)
end

GDOptimizer(conf=get_empty_config()) = GDOptimizer(0, 0, conf)

"""
    has_weight_tags(conf) -> Bool

Return whether band or k-point weights (`wE`, `wk`, `wE_<n>`, `wk_<n>`) are set in the `Optimizer` block.
"""
function has_weight_tags(conf)
    haskey(conf.blocks, "Optimizer") || return false
    return any(key -> key ∈ ("we", "wk") || startswith(key, "we_") || startswith(key, "wk_"), lowercase.(keys(conf.blocks["Optimizer"])))
end

"""
    weight_warning(dl, Nε, Nk, val_weights, conf; write_output=true)

Warn (and write a line to `hamster.out` if `write_output`) if band or k-point weights are set but cannot be applied:
in fits to Hamiltonian data, if the training structures differ in the number of bands or k-points (`Nε = Nk = 0`),
and, with `val_weights = true`, if the validation data has another number of bands or k-points than the training data.
"""
function weight_warning(dl, Nε, Nk, val_weights, conf; write_output=true)
    has_weight_tags(conf) || return nothing
    message = if eltype(dl.train_data) <: HrData
        "Band and k-point weights (wE, wk) are ignored in fits to Hamiltonian data."
    elseif Nε == 0 && Nk == 0
        "Band and k-point weights (wE, wk) are ignored: the training structures differ in the number of bands or k-points."
    elseif get_val_weights(conf) && get_validate(conf) && !val_weights
        "Band and k-point weights (wE, wk) are not used for validation (val_weights = true): the validation data has another number of bands or k-points than the training data."
    else
        return nothing
    end
    @warn message
    write_output && append_output_line("Warning: " * message)
    return nothing
end

