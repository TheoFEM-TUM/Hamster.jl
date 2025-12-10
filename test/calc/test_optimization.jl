@testset "Optimization workflow for TB" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.15
    @test prof.L_val[end] < 0.15
    rm("hamster.out"); rm("params.dat")
end

@testset "Optimization workflow for TB+SOC" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_soc"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_soc"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.20
    @test mean(prof.L_train[:, end]) < mean(prof.L_train[:, 1])
    rm("hamster.out"); rm("params.dat")
end

@testset "Optimization workflow for ML+TB" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_ml"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))

    prof = Hamster.main(comm, conf, rank=rank)
    println("Errors", prof.L_train[1, :])
    println("Errors", prof.L_val[:])
    @test mean(prof.L_train[:, end]) < 0.1
    @test prof.L_val[end] < 0.3 # includes all bands
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
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
    println("Errors", prof.L_train[1, :])
    println("Errors", prof.L_val[:])
    @test mean(prof.L_train[:, end]) < 0.1
    @test prof.L_val[end] < 0.3 # includes all bands
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
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
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
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
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
end

rm("hamster_out.h5")