@testset "Standard PC calculation" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_std"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "kpoints", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))

    prof = Hamster.main(comm, conf, rank=rank)
    Es_tb = read_from_file("Es.dat")
    _, Es_dft, _ = Hamster.read_eigenval(conf("kpoints"), 8)
    @test mean(abs.(Es_tb .- Es_dft)) < 0.3
    rm("hamster.out"); rm("config_inds.dat"); rm("Es.dat")
end

@testset "Standard MD calculation" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_std_md"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
    set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "kpoints", joinpath(path, "eigenval.h5"))
    set_value!(conf, "init_params", joinpath(path, "params.dat"))

    prof = Hamster.main(comm, conf, rank=rank)
    Es_dft = h5read(joinpath(path, "eigenval.h5"), "eigenvalues")
    inds = Int.(read_from_file("config_inds.dat"))
    for ind in inds
        Es_tb = read_from_file("tmp/Es$ind.dat")
        @test mean(abs.(Es_dft[:, :, ind] .- Es_tb)) < 0.3
        @test mean(abs.(Es_dft[:, :, ind] .- Es_tb)) < 0.3
    end

    rm("hamster.out"); rm("config_inds.dat"); rm("tmp", recursive=true); rm("Es.dat"); rm("vs.dat")
end