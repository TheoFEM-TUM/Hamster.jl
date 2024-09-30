gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"
cspbbr_poscar = string(@__DIR__) * "/../strc/test_files/POSCAR_CsPbBr3"

 
@testset "Rllm GaAs" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "interpolate_rllm", true)
    set_value!(conf, "rllm_file", string(@__DIR__)*"/test_files/rllm.dat")
    set_value!(conf, "verbosity", 0)

    strc_gaas = Structure(conf, poscar_path=gaas_poscar)
    orbitals_gaas = Hamster.get_orbitals(strc_gaas, conf)

    overlaps_gaas = Hamster.get_overlaps(strc_gaas.ions, orbitals_gaas, conf)

    rllm = Hamster.get_rllm(overlaps_gaas, conf)
    rllm_correct = Hamster.read_rllm(string(@__DIR__)*"/test_files/rllm_true.dat")

    is_correct = Bool[]
    for key in keys(rllm)
        f1 = rllm[key]
        f2 = rllm_correct[key]
        xs = rand(100) .* 5.
        is_correct = mean(abs.(f1.(xs) .- f2.(xs) )) < 0.001
    end
    @test all(is_correct)
    rm(string(@__DIR__)*"/test_files/rllm.dat")
end