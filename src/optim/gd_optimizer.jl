struct GDOptimizer
    loss :: Loss
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
"""
function GDOptimizer(Nε, Nk, conf=get_empty_config(); lr=get_lr(conf), Niter=get_niter(conf))
    loss = Nε == 0 && Nk == 0 ? Loss(conf) : Loss(Nε, Nk, conf)
    reg = Regularization(conf)
    adam = Adam(lr)
    return GDOptimizer(loss, reg, adam, Niter)
end

GDOptimizer(conf=get_empty_config()) = GDOptimizer(0, 0, conf)

