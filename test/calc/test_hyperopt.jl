@testset "Random search hyperparameter optimization" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test std(prof.L_train) > 0
    @test std(prof.timings) > 0
    rm("hamster.out"); rm("train_config_inds.dat"); rm("val_config_inds.dat"); rm("params.dat")
    rm("L_val.dat"); rm("L_train.dat"); rm("rllm.dat"); rm("hyperopt_out.h5")
end

@testset "Grid search hyperparameter optimization" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "stepsizes", "HyperOpt", "10 10")
    set_value!(conf, "mode", "HyperOpt", "grid")

    prof = Hamster.main(comm, conf, rank=rank)
    @test std(prof.L_train) > 0
    @test std(prof.timings) > 0
    @test h5read("hyperopt_out.h5", "param_values") == [5.0 15.0 5.0; 5.0 5.0 15.0]
    rm("hamster.out"); rm("train_config_inds.dat"); rm("val_config_inds.dat"); rm("params.dat")
    rm("L_val.dat"); rm("L_train.dat"); rm("rllm.dat"); rm("hyperopt_out.h5")
end