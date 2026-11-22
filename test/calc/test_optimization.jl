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
    # The linear algebra library is listed at the top of hamster.out
    @test any(line -> occursin("Linear algebra (BLAS/LAPACK): OpenBLAS", line), readlines("hamster.out")[1:30])
    rm("hamster.out"); rm("params.dat")
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
    set_value!(conf, "sp_mode", false)

    prof = Hamster.main(comm, conf, rank=rank)
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

    # The ML data points are selected randomly (kmeans, sampling within clusters): seed, so that the loss does not
    # depend on the tests that ran before
    Hamster.Random.seed!(1234)
    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.20
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
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
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
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
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
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
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
end

include("fit_helpers.jl")
@testset "Validation set of MD fits" begin
    path = joinpath(@__DIR__, "test_files")
    conf = validated_md_fit_config(path)
    set_value!(conf, "Nconf", "Supercell", 2)
    @test_throws "validation set is empty" Hamster.run_calculation(Val{:optimization}(), comm, conf, write_output=false)

    mktempdir() do dir
        cd(dir) do
            conf = validated_md_fit_config(path)
            set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
            set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
            set_value!(conf, "val_mode", "Optimizer", "pc")
            prof = Hamster.main(comm, conf, rank=rank)
            @test all(>(0), prof.L_train)
            @test all(>(0), prof.L_val)
        end
    end
end

@testset "Validation structures from a trajectory of their own" begin
    path = joinpath(@__DIR__, "test_files")
    lattice, positions = Hamster.read_xdatcar(joinpath(path, "XDATCAR"), frac=true)
    mktempdir() do dir
        # configurations 8 to 10 of the trajectory and of the data, as files of their own
        h5open(joinpath(dir, "val_strc.h5"), "w") do f
            f["positions"] = positions[:, :, 8:10]
            f["lattice"] = ndims(lattice) == 3 ? lattice[:, :, 8:10] : lattice
        end
        h5open(joinpath(dir, "val_data.h5"), "w") do f
            f["eigenvalues"] = h5read(joinpath(path, "eigenval.h5"), "eigenvalues")[:, :, 8:10]
            f["kpoints"] = h5read(joinpath(path, "eigenval.h5"), "kpoints")
        end
        losses = map(["same", "own"]) do case
            mkdir(joinpath(dir, case))
            cd(joinpath(dir, case)) do
                conf = validated_md_fit_config(path)
                set_value!(conf, "config_inds", "Supercell", "1 2 3 4")
                if case == "same"
                    set_value!(conf, "val_config_inds", "Supercell", "8 9 10")
                else
                    set_value!(conf, "val_xdatcar", "Optimizer", joinpath(dir, "val_strc.h5"))
                    set_value!(conf, "val_data", "Optimizer", joinpath(dir, "val_data.h5"))
                end
                prof = Hamster.main(comm, conf, rank=rank)
                @test h5read("hamster_out.h5", "val_config_inds") == (case == "same" ? [8, 9, 10] : [1, 2, 3])
                prof.L_val
            end
        end
        @test all(>(0), losses[2]) && losses[2] ≈ losses[1]
    end
end

@testset "More batches than training structures" begin
    @test Hamster.get_local_batches(5, 3) == [1:2, 3:4, 5:5]
    @test Hamster.get_local_batches(2, 3) == [1:1, 2:2, 1:0]
    @test Hamster.get_local_batches(0, 2) == [1:0, 1:0]
    @test Hamster.get_number_of_batches(2, 5, comm) == 2
    @test_throws ErrorException Hamster.get_number_of_batches(0, 5, comm)

    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_md"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_md"))
    set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
    set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "eigenval.h5"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "nbatch", "Supercell", 5)
    set_value!(conf, "niter", "Optimizer", 2)

    prof = @test_logs (:warn, r"nbatch = 5 is larger") match_mode=:any Hamster.main(comm, conf, rank=rank)
    @test size(prof.L_train) == (2, 2)
    @test all(>(0), prof.L_train)
    output = read("hamster.out", String)
    @test occursin("Warning: nbatch = 5 is larger", output)
    @test occursin(r"The run gave \d+ warnings?, see", output)
    rm("hamster.out"); rm("params.dat"); rm("ml_params.dat")
end

rm("hamster_out.h5")
@testset "Sparse diagonalization is rejected in fits" begin
    conf = get_empty_config()
    set_value!(conf, "lr", "Optimizer", 0.1)
    set_value!(conf, "sp_diag", true)
    @test_throws ErrorException Hamster.run_calculation(Val{:optimization}(), comm, conf, write_output=false)
end
