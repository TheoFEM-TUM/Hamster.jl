@testset "Random search hyperparameter optimization" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test std(prof.L_train) > 0
    @test std(prof.timings) > 0
    rm("hamster.out"); rm("rllm.dat"); rm("hamster_out.h5")
end

@testset "Grid search hyperparameter optimization" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "lowerbounds", "HyperOpt", 5)
    set_value!(conf, "upperbounds", "HyperOpt", 7)
    set_value!(conf, "params", "HyperOpt", "rcut")
    set_value!(conf, "stepsizes", "HyperOpt", "1")
    set_value!(conf, "mode", "HyperOpt", "grid")
    set_value!(conf, "niter", "HyperOpt", 1)

    prof = Hamster.main(comm, conf, rank=rank)
    @test std(prof.L_train) > 0
    @test std(prof.timings) > 0
    @test h5read("hamster_out.h5", "rcut") == [5.0, 6.0, 7.0]
    rm("hamster.out"); rm("rllm.dat"); rm("hamster_out.h5")
end

@testset "TPE hyperparameter optimization" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "mode", "HyperOpt", "tpe")

    prof = Hamster.main(comm, conf, rank=rank)
    @test std(prof.L_train) > 0
    @test std(prof.timings) > 0
    rm("hamster.out"); rm("rllm.dat"); rm("hamster_out.h5")
end