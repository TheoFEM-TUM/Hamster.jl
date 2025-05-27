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

