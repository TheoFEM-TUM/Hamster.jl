"""
    validated_md_fit_config(path)

Config of a validated TB fit to the GaAs MD data in `path`: 4 training structures and 1 validation structure.
"""
function validated_md_fit_config(path)
    conf = get_config(filename = joinpath(path, "hconf_md"))
    set_value!(conf, "model", "ML", false)
    set_value!(conf, "poscar", joinpath(path, "POSCAR_md"))
    set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
    set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "seed", 1)
    set_value!(conf, "Nconf", "Supercell", 4)
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "eigenval.h5"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "eigenval.h5"))
    set_value!(conf, "val_mode", "Optimizer", "md")
    set_value!(conf, "val_ratio", "Optimizer", 0.25)
    set_value!(conf, "niter", "Optimizer", 2)
    return conf
end
