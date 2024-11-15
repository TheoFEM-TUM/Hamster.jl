struct GDOptimizer
    loss :: Loss
    reg :: Regularization
    optim :: Adam
    Niter :: Int64
end

function GDOptimizer(Nε, Nk, conf=get_empty_config(); lr=get_lr(conf), Niter=get_niter(conf))
    loss = Nε == 0 && Nk == 0 ? Loss(conf) : Loss(Nε, Nk, conf)
    reg = Regularization(conf)
    optim = Adam(lr)
    return GDOptimizer(loss, reg, optim, Niter)
end

GDOptimizer(conf=get_empty_config()) = GDOptimizer(0, 0, conf)

