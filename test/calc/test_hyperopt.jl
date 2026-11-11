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
    set_value!(conf, "upperbounds", "HyperOpt", 6)
    set_value!(conf, "params", "HyperOpt", "rcut")
    set_value!(conf, "stepsizes", "HyperOpt", "1")
    set_value!(conf, "mode", "HyperOpt", "grid")
    set_value!(conf, "niter", "HyperOpt", 1)

    prof = Hamster.main(comm, conf, rank=rank)
    @test std(prof.L_train) > 0
    @test std(prof.timings) > 0
    @test h5read("hamster_out.h5", "rcut") == [5.0, 6.0]
    rm("hamster.out"); rm("rllm.dat"); rm("hamster_out.h5")
end

@testset "Hyperparameter optimization objective" begin
    prof = Hamster.HamsterProfiler(3, Niter=4, Nbatch=1)
    prof.L_train .= [4.0 3.0 2.0 1.0]
    prof.L_train_BG .= [0.4 0.1 0.3 0.2]
    # Iterations 3 and 4 were not validated (e.g., early stopping) and must be skipped
    prof.L_val[1:2] .= [5.0, 6.0]
    prof.L_val_MAE[1:2] .= [0.7, 0.8]
    prof.L_val_BG[1:2] .= [0.6, 0.5]

    @test Hamster.get_hyperopt_loss(prof, "loss") == 1.0
    @test Hamster.get_hyperopt_loss(prof, "bg") == 0.1
    @test Hamster.get_hyperopt_loss(prof, "loss", validate=true) == 5.0
    @test Hamster.get_hyperopt_loss(prof, "mae", validate=true) == 0.7
    @test Hamster.get_hyperopt_loss(prof, "bg", validate=true) == 0.5
    @test Hamster.get_hyperopt_loss(Hamster.HamsterProfiler(3, Niter=2, Nbatch=1), "bg", validate=true) == Inf
    @test_throws ErrorException Hamster.get_hyperopt_loss(prof, "gap")
end

@testset "Crashing trials do not abort the hyperparameter optimization" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "does_not_exist"))

    prof = with_logger(NullLogger()) do
        Hamster.main(comm, conf, rank=rank)
    end
    @test size(prof.L_train) == (1, 2)
    @test all(isinf, prof.L_train)
    @test all(>(0), prof.timings)
    @test size(h5read("hamster_out.h5", "Ga_alpha")) == (2,)
    for file in ("hamster.out", "rllm.dat", "hamster_out.h5"); rm(file, force=true); end
end

@testset "Band gap hyperparameter optimization with validation" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "metric", "HyperOpt", "BG")

    prof = Hamster.main(comm, conf, rank=rank)
    @test all(L -> 0 < L < Inf, prof.L_train)
    @test std(prof.L_train) > 0
    rm("hamster.out"); rm("rllm.dat"); rm("hamster_out.h5")
end

@testset "TPE hyperparameter optimization" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "mode", "HyperOpt", "tpe")
    set_value!(conf, "log_modes", "HyperOpt", ["uni", "log"])

    prof = Hamster.main(comm, conf, rank=rank)
    @test std(prof.L_train) > 0
    @test std(prof.timings) > 0
    rm("hamster.out"); rm("rllm.dat"); rm("hamster_out.h5")
end