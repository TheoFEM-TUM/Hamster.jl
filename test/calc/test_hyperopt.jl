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

@testset "Hyperparameter labels and settings" begin
    @test Hamster.hyperopt_tag("rcut") == ("rcut", "Options")
    @test Hamster.hyperopt_tag("grid_size") == ("grid_size", "Options")
    @test Hamster.hyperopt_tag("ML_rcut") == ("rcut", "ML")
    @test Hamster.hyperopt_tag("Optimizer_val_ratio") == ("val_ratio", "Optimizer")
    @test Hamster.hyperopt_tag("val_ratio") == ("val_ratio", "Optimizer")
    @test Hamster.hyperopt_tag("lr") == ("lr", "Optimizer")
    @test Hamster.hyperopt_tag("niter") == ("niter", "Optimizer")
    @test Hamster.hyperopt_tag("Ga_alpha") == ("alpha", "Ga")
    @test Hamster.hyperopt_tag("wk_1") == ("wk_1", "Optimizer")
    @test_throws "per species" Hamster.hyperopt_tag("n")
    @test_throws "several blocks" Hamster.hyperopt_tag("alpha")
    @test_throws "Unknown hyperparameter" Hamster.hyperopt_tag("ML_foo")

    @test isnothing(Hamster.check_hyperopt_settings(["rcut", "lr"], [5., 0.1], [6., 0.5], [1., 0.1], "random"))
    @test_throws "lowerbounds" Hamster.check_hyperopt_settings(["rcut", "lr"], [5.], [6., 0.5], [1., 0.1], "random")
    @test_throws "stepsizes" Hamster.check_hyperopt_settings(["rcut"], [5.], [6.], [1., 0.1], "grid")
    @test_throws "larger than the upper bound" Hamster.check_hyperopt_settings(["rcut"], [6.], [5.], [1.], "grid")
    @test_throws "positive" Hamster.check_hyperopt_settings(["rcut"], [5.], [6.], [0.], "grid")
    @test_throws "Unknown mode" Hamster.check_hyperopt_settings(["rcut"], [5.], [6.], [1.], "bayes")
    @test Hamster.split_depends_on([("rcut", "Options"), ("Nconf", "Supercell")])
    @test Hamster.split_depends_on([("val_ratio", "Optimizer")])
    @test !Hamster.split_depends_on([("rcut", "Options"), ("lr", "Optimizer")])
end

@testset "Hyperparameters of other blocks and the full grid" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_hyperopt"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "params", "HyperOpt", "lr")
    set_value!(conf, "lowerbounds", "HyperOpt", 0.1)
    set_value!(conf, "upperbounds", "HyperOpt", 0.5)
    set_value!(conf, "stepsizes", "HyperOpt", 0.4)
    set_value!(conf, "mode", "HyperOpt", "grid")
    set_value!(conf, "niter", "HyperOpt", 3)

    prof = Hamster.main(comm, conf, rank=rank)
    @test size(prof.L_train) == (1, 2)
    @test prof.L_train[1, 1] != prof.L_train[1, 2]
    h5open("hamster_out.h5") do file
        @test read(file["lr"]) ≈ [0.1, 0.5]
        @test read(file["params"]) == ["lr"]
        @test read(file["L_fit"]) == vec(read(file["L_train"]))
        @test read(attributes(file)["evaluated"]) == 2
    end
    rm("hamster.out"); rm("rllm.dat"); rm("hamster_out.h5")
end

function hyperopt_md_config(path)
    conf = get_config(filename = joinpath(path, "hconf_md"))
    delete!(conf.blocks, "ML")
    set_value!(conf, "poscar", joinpath(path, "POSCAR_md"))
    set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
    set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "eigenval.h5"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "eigenval.h5"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "niter", "Optimizer", 3)
    set_value!(conf, "val_mode", "Optimizer", "md")
    set_value!(conf, "val_ratio", "Optimizer", 0.5)
    set_value!(conf, "params", "HyperOpt", "printeachiter")
    set_value!(conf, "lowerbounds", "HyperOpt", 1)
    set_value!(conf, "upperbounds", "HyperOpt", 4)
    set_value!(conf, "stepsizes", "HyperOpt", 1)
    set_value!(conf, "mode", "HyperOpt", "grid")
    return conf
end

@testset "Same configurations for all candidates" begin
    conf = hyperopt_md_config(joinpath(@__DIR__, "test_files"))
    prof = Hamster.main(comm, conf, rank=rank)
    @test all(prof.L_train .> 0)
    @test all(prof.L_train[1, :] .≈ prof.L_train[1, 1])
    @test vec(prof.L_train) == prof.L_val
    h5open("hamster_out.h5") do file
        @test length(read(file["train_config_inds"])) == 2
        @test length(read(file["val_config_inds"])) == 1
        @test all(read(file["L_fit"]) .≈ read(file["L_fit"])[1])
        @test read(file["L_val"]) == prof.L_val
    end
    rm("hamster.out"); rm("hamster_out.h5")
end

@testset "Validation loss of candidates with valeachiter > 1" begin
    conf = hyperopt_md_config(joinpath(@__DIR__, "test_files"))
    set_value!(conf, "valeachiter", "Optimizer", 2)
    set_value!(conf, "upperbounds", "HyperOpt", 2)
    prof = Hamster.main(comm, conf, rank=rank)
    @test all(prof.L_train .> 0)
    @test vec(prof.L_train) == prof.L_val
    rm("hamster.out"); rm("hamster_out.h5")
end

@testset "Hyperopt results after every candidate" begin
    prof = Hamster.HamsterProfiler(1, Niter=3, Nbatch=1, Nparams=2)
    prof.L_train[1, 1] = 0.5; prof.L_val[1] = 0.5; prof.param_values[:, 1] = [5., 6.]
    L_fit = [0.4, 0., 0.]
    mktempdir() do dir
        file = joinpath(dir, "hamster_out.h5")
        Hamster.write_hyperopt_results(prof, L_fit, ["rcut", "Ga_alpha"], 1, filename=file)
        prof.L_train[1, 2] = 0.3; prof.param_values[:, 2] = [7., 8.]; L_fit[2] = 0.2
        Hamster.write_hyperopt_results(prof, L_fit, ["rcut", "Ga_alpha"], 2, filename=file)
        h5open(file) do f
            @test read(f["L_train"]) == [0.5 0.3 0.]
            @test read(f["L_fit"]) == [0.4, 0.2, 0.]
            @test read(f["rcut"]) == [5., 7., 0.]
            @test read(f["Ga_alpha"]) == [6., 8., 0.]
            @test read(attributes(f)["evaluated"]) == 2
        end
    end
end
