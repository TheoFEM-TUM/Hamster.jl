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