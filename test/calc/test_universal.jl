include("universal_helpers.jl")

@testset "Simple Universal Model" begin
    path = joinpath(@__DIR__, "test_files")
    strc_file = joinpath(path, "strc_file.h5") 
    eig_file = joinpath(path, "eig_file.h5")
    if isfile(strc_file); rm(strc_file); end
    if isfile(eig_file); rm(eig_file); end
    
    kp_gaas, Es_gaas, _ = Hamster.read_eigenval(joinpath(path, "EIGENVAL_gaas"))
    kp_cspbbr3, Es_cspbbr3, _ = Hamster.read_eigenval(joinpath(path, "EIGENVAL_cspbbr3"))

    # Generate dataset for structures
    write_universal_strc_file(strc_file)

    # Generate dataset for eigenvalues
    h5open(eig_file, "cw") do f
        g1 = create_group(f, "cspbbr3")
        write(g1, "kpoints", kp_cspbbr3)
        write(g1, "eigenvalues", reshape(Es_cspbbr3[14:end, :], 17, 84, 1))

        g2 = create_group(f, "gaas")
        write(g2, "kpoints", kp_gaas)
        write(g2, "eigenvalues", reshape(Es_gaas, 48, 56, 1))
    end

    # Run calculation
    conf = get_config(filename = joinpath(path, "hconf_universal"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_universal.dat"))
    set_value!(conf, "train_data", "Optimizer", eig_file)
    set_value!(conf, "xdatcar", "Supercell", strc_file)
    #set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.2
    rm("hamster.out"); rm("hamster_out.h5"); rm("params.dat"); rm(strc_file); rm(eig_file)
end

@testset "Standard run with several systems" begin
    mktempdir() do dir
        cd(dir) do
            strc_file = joinpath(dir, "strc_file.h5")
            write_universal_strc_file(strc_file)
            conf = universal_standard_config(strc_file)
            Hamster.TEMP_DIR[] = "run_tmp"
            mkdir("run_tmp")
            try
                Hamster.run_calculation(Val{:standard}(), comm, conf, rank=rank, write_output=false)
                Hamster.collapse_eigenvalue_files(conf)
            finally
                Hamster.TEMP_DIR[] = "tmp"
            end
            @test !isfile("Es.dat")
            @test size(read_from_file("Es_cspbbr3.dat")) == (14, 1, 1)
            @test size(read_from_file("Es_gaas.dat")) == (8, 1, 1)
            labels_cspbbr3 = h5read("ham.h5", "basis_cspbbr3")
            @test length(labels_cspbbr3) == 14 && "Cs-s" in labels_cspbbr3 && "Br-px" in labels_cspbbr3
            @test h5read("ham.h5", "basis_gaas") == vcat(fill("Ga-sp3dr2", 4), fill("As-sp3dr2", 4))
            @test h5open(file -> haskey(file, "Hr_cspbbr3_1") && haskey(file, "Hr_gaas_1"), "ham.h5")
        end
    end
end

@testset "Universal fit with weights of the eigenvalue data" begin
    mktempdir() do dir
        cd(dir) do
            path = joinpath(@__DIR__, "test_files")
            strc_file = joinpath(dir, "strc_file.h5")
            write_universal_strc_file(strc_file)
            kp_gaas, Es_gaas, _ = Hamster.read_eigenval(joinpath(path, "EIGENVAL_gaas"))
            kp_cspbbr3, Es_cspbbr3, _ = Hamster.read_eigenval(joinpath(path, "EIGENVAL_cspbbr3"))
            function fit(scale)
                eig_file = joinpath(dir, "eig_file.h5")
                isfile(eig_file) && rm(eig_file)
                h5open(eig_file, "w") do f
                    for (name, kp, Es) in (("cspbbr3", kp_cspbbr3, reshape(Es_cspbbr3[14:end, :], 17, 84, 1)),
                                           ("gaas", kp_gaas, reshape(Es_gaas, 48, 56, 1)))
                        g = create_group(f, name)
                        g["kpoints"] = kp; g["eigenvalues"] = Es
                        isnothing(scale) || (g["weights"] = fill(scale, size(Es, 1)))
                    end
                end
                conf = get_config(filename=joinpath(path, "hconf_universal"))
                set_value!(conf, "rllm_file", joinpath(path, "rllm_universal.dat"))
                set_value!(conf, "train_data", "Optimizer", eig_file)
                set_value!(conf, "xdatcar", "Supercell", strc_file)
                set_value!(conf, "niter", "Optimizer", 2)
                return Hamster.main(comm, conf, rank=rank).L_train
            end
            L = fit(nothing)
            @test fit(1.0) ≈ L
            @test fit(2.0)[:, 1] ≈ 2 .* L[:, 1]
        end
    end
end
