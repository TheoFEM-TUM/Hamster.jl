# Writing the parameters during a fit and continuing a fit from a checkpoint. The checkpoint folder contains the
# parameter files of the latest iteration (params.dat, ml_params.dat) and optimizer.h5 (iteration, Adam moments, losses,
# configuration indices); params.dat in the run folder contains the parameters with the lowest validation loss so far
# (the latest parameters without validation).

cosine_lr(iter, Niter, lr, lr_min) = lr_min + 0.5 * (lr - lr_min) * (1 + cos(π * iter / Niter))

"""
    write_params_atomically(ham, conf; dir=".")

Write the parameter files of all models to `dir`. They are written to a temporary folder first and then renamed, so that
an interrupted run never leaves a partially written parameter file.
"""
function write_params_atomically(ham, conf; dir=".")
    mkpath(dir)
    tmp = mktempdir(dir; prefix=".params_", cleanup=false)
    try
        cd(() -> write_params(ham, conf), tmp)
        for file in readdir(tmp)
            Base.Filesystem.rename(joinpath(tmp, file), joinpath(dir, file))
        end
    finally
        rm(tmp, recursive=true, force=true)
    end
end

"""
    write_checkpoint(ham, optim, prof, iter, best_L_val, config_inds, conf; dir=get_checkpoint(conf))

Write the checkpoint of iteration `iter`: parameter files and optimizer state. The new checkpoint replaces the old one
only when it is complete (the old one is kept as `<dir>.old` until then).
"""
function write_checkpoint(ham, optim, prof, iter, best_L_val, config_inds, conf; dir=get_checkpoint(conf))
    dir = rstrip(dir, '/')
    tmp = dir * ".tmp"
    rm(tmp, recursive=true, force=true)
    write_params_atomically(ham, conf, dir=tmp)
    h5open(joinpath(tmp, "optimizer.h5"), "w") do file
        file["iter"] = iter
        file["best_L_val"] = best_L_val
        file["L_train"] = prof.L_train[:, 1:iter]
        file["L_val"] = prof.L_val[1:iter]
        train_inds, val_inds = config_inds
        file["train_config_inds"] = train_inds
        file["val_config_inds"] = val_inds
        for (m, model) in enumerate(ham.models)
            params = get_params(model)
            haskey(optim.adam.state, params) || continue
            mt, vt, βp = optim.adam.state[params]
            file["adam/$m/mt"] = mt
            file["adam/$m/vt"] = vt
            file["adam/$m/betap"] = βp
        end
    end
    isdir(dir) && (rm(dir * ".old", recursive=true, force=true); Base.Filesystem.rename(dir, dir * ".old"))
    Base.Filesystem.rename(tmp, dir)
    rm(dir * ".old", recursive=true, force=true)
end

"""
    checkpoint_dir(conf; dir=get_checkpoint(conf))

Folder of the latest complete checkpoint (`<dir>.old` if the run stopped while the checkpoint was replaced).
"""
function checkpoint_dir(conf; dir=get_checkpoint(conf))
    dir = rstrip(dir, '/')
    for candidate in (dir, dir * ".old")
        isfile(joinpath(candidate, "optimizer.h5")) && return candidate
    end
    error("restart = true, but there is no checkpoint in $dir.")
end

"""
    prepare_restart!(conf, dir; Nsystems=1)

Set the initial parameters of all models (and the configuration indices of a single MD system) to the checkpoint in `dir`.
"""
function prepare_restart!(conf, dir; Nsystems=length(get_systems(conf)))
    params_file = joinpath(dir, "params.dat")
    isfile(params_file) && set_value!(conf, "init_params", params_file)
    ml_file = joinpath(dir, basename(get_ml_filename(conf)) * ".dat")
    if get_ml_model(conf) && isfile(ml_file)
        set_value!(conf, "init_params", "ML", ml_file)
        get_ml_mode(conf) == "expand" && set_value!(conf, "mode", "ML", "refit")
    end
    haskey(conf, "SOC") && set_value!(conf, "init_params", "SOC", params_file)
    haskey(conf, "Ewald") && set_value!(conf, "init_params", "Ewald", params_file)
    if Nsystems == 1 && haskey(conf, "Supercell")
        set_value!(conf, "config_inds", "Supercell", joinpath(dir, "optimizer.h5"))
        set_value!(conf, "val_config_inds", "Supercell", joinpath(dir, "optimizer.h5"))
    end
end

"""
    restore_optimizer!(optim, prof, ham, dir, conf) -> (iter, best_L_val)

Restore the Adam moments, the learning rate and the loss history from the checkpoint in `dir`; returns the last iteration
of the checkpoint and the lowest validation loss so far.
"""
function restore_optimizer!(optim, prof, ham, dir, conf; lr=get_lr(conf), lr_min=get_lr_min(conf))
    iter, best_L_val = h5open(joinpath(dir, "optimizer.h5"), "r") do file
        for (m, model) in enumerate(ham.models)
            haskey(file, "adam/$m") || continue
            params = get_params(model)
            mt, vt = read(file["adam/$m/mt"]), read(file["adam/$m/vt"])
            size(mt) == size(params) || error("The checkpoint in $dir does not match model $m (parameter size).")
            optim.adam.state[params] = (mt, vt, read(file["adam/$m/betap"]))
        end
        L_train, L_val = read(file["L_train"]), read(file["L_val"])
        n = min(size(L_train, 2), size(prof.L_train, 2))
        nb = min(size(L_train, 1), size(prof.L_train, 1))
        prof.L_train[1:nb, 1:n] .= L_train[1:nb, 1:n]
        prof.L_val[1:min(length(L_val), length(prof.L_val))] .= L_val[1:min(length(L_val), length(prof.L_val))]
        read(file["iter"]), read(file["best_L_val"])
    end
    optim.adam.eta = cosine_lr(iter, optim.Niter, lr, lr_min)
    return iter, best_L_val
end
