gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"

@testset "Basis GaAs" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", string(@__DIR__)*"/test_files/rllm_true.dat")
    set_value!(conf, "verbosity", 0)

    strc = Structure(conf, poscar_path=gaas_poscar)
    basis = Basis(strc, conf)

    @test length(basis) == 8
    @test size(basis) == [4, 4]
    @test nparams(basis) == length(basis.parameters)
end
