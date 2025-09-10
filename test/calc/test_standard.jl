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
    rm("hamster.out"); rm("Es.dat")
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
    Es_tb = read_from_file("Es.dat")
    inds = h5read("hamster_out.h5", "config_inds.dat")
    for (i, ind) in enumerate(inds)
        @test mean(abs.(Es_dft[:, :, ind] .- Es_tb[:, :, i])) < 0.3
    end

    mm("hamster.out"); rm("Es.dat")
end