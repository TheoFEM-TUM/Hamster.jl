@testset "Optimization workflow for TB" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "sp_mode", false)

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.15
    @test prof.L_val[end] < 0.15
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true)
end

@testset "Optimization workflow for TB+SOC" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_soc"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_soc"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "sp_mode", false)

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.20
    @test mean(prof.L_train[:, end]) < mean(prof.L_train[:, 1])
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true)
end

@testset "Optimization workflow for ML+TB" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_ml"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "sp_mode", false)

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.1
    @test prof.L_val[end] < 0.3 # includes all bands
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

@testset "Optimization workflow for ML+TB (sparse)" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_ml"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "sp_mode", true)
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.1
    @test prof.L_val[end] < 0.3 # includes all bands
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

@testset "Optimization workflow for ML+TB+SOC (sparse)" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_cspbbr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "sp_mode", true)
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_cspbbr3_soc"))
    set_value!(conf, "init_params", joinpath(path, "params_cspbbr3.dat"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.20
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

@testset "Optimization workflow for ML+TB (sparse) using sim_mat" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_ml"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "sp_mode", true)
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "sim_mat", "ML", true)

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.1
    @test prof.L_val[end] < 0.5 # includes all bands
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

@testset "No update for ML+TB (sparse) using sim_mat" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_ml"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "sp_mode", true)
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "sim_mat", "ML", true)
    set_value!(conf, "update", "ML", false)
    set_value!(conf, "update_tb", "Optimizer", false)
    set_value!(conf, "niter", "Optimizer", 2)

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 1
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

@testset "Optimization workflow for ML+TB+SOC (sparse) using sim_mat" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_cspbbr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "sp_mode", true)
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_cspbbr3_soc"))
    set_value!(conf, "init_params", joinpath(path, "params_cspbbr3.dat"))
    set_value!(conf, "sim_mat", "ML", true)
    set_value!(conf, "key_dims", "ML", "1 2")
    set_value!(conf, "init_params", "ML", joinpath(path, "ml_params_cspbbr3.dat"))
    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.20
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

@testset "Optimization for MD data (sparse)" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_md"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_md"))
    set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
    set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "eigenval.h5"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.15
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

@testset "TB optimization with several batches" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_md"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_md"))
    set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
    set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "eigenval.h5"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "Nconf", "Supercell", 4)
    set_value!(conf, "nbatch", "Supercell", 2)
    set_value!(conf, "niter", "Optimizer", 2)

    prof = Hamster.main(comm, conf, rank=rank)
    @test size(prof.L_train) == (2, 2)
    @test all(isfinite, prof.L_train)
    rm("hamster.out"); rm("params.dat"); rm("checkpoint", recursive=true, force=true); rm("ml_params.dat")
end

rm("hamster_out.h5")
@testset "Checkpoint and restart of a fit" begin
    path = joinpath(@__DIR__, "test_files")
    function fit_config(niter; restart=false)
        conf = get_config(filename = joinpath(path, "hconf_ml"))
        set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
        set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
        set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
        set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
        set_value!(conf, "init_params", joinpath(path, "params.dat"))
        set_value!(conf, "sp_mode", false)
        set_value!(conf, "seed", 1)
        set_value!(conf, "niter", "Optimizer", niter)
        set_value!(conf, "restart", "Optimizer", restart)
        return conf
    end
    mktempdir() do dir
        cd(dir) do
            prof_ref = Hamster.main(comm, fit_config(5), rank=rank)
            params_ref = read("checkpoint/params.dat", String)
            ml_ref = read("checkpoint/ml_params.dat", String)
            @test isfile("params.dat") && isfile("checkpoint/optimizer.h5")
            @test h5read("checkpoint/optimizer.h5", "iter") == 5
            # params.dat: parameters with the lowest validation loss
            @test h5read("checkpoint/optimizer.h5", "best_L_val") ≈ minimum(prof_ref.L_val)
            rm("checkpoint", recursive=true)

            # The same fit, stopped after 3 iterations and continued: identical parameters and losses (constant lr)
            Hamster.main(comm, fit_config(3), rank=rank)
            @test h5read("checkpoint/optimizer.h5", "iter") == 3
            prof = Hamster.main(comm, fit_config(5, restart=true), rank=rank)
            @test read("checkpoint/params.dat", String) == params_ref
            @test read("checkpoint/ml_params.dat", String) == ml_ref
            @test prof.L_train ≈ prof_ref.L_train
            @test prof.L_val ≈ prof_ref.L_val
            @test !isdir("checkpoint.old") && !isdir("checkpoint.tmp")

            @test_throws ErrorException Hamster.checkpoint_dir(fit_config(5), dir="no_checkpoint")
        end
    end
end
