gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"
cspbbr_poscar = string(@__DIR__) * "/../strc/test_files/POSCAR_CsPbBr3"

@testset "GaAs overlaps" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "interpolate_rllm", false)
    
    strc_gaas = Structure(conf, poscar_path=gaas_poscar)
    orbitals_gaas = Hamster.get_orbitals(strc_gaas, conf)

    overlaps_gaas = Hamster.get_overlaps(strc_gaas.ions, orbitals_gaas, conf)
    @test length(overlaps_gaas) == 34
    @test sort(string.(overlaps_gaas, apply_oc=true)) == sort(["Ga+Ga_ssσ", "Ga+Ga_spσ", "Ga+Ga_sdσ", "Ga+Ga_ppπ", "Ga+Ga_pdπ", "Ga+Ga_ppσ", "Ga+Ga_pdσ", "Ga+Ga_ddπ", "Ga+Ga_ddδ", "Ga+Ga_ddσ", "Ga+As_ssσ", "Ga+As_spσ", "Ga+As_sdσ", "Ga+As_ppπ", "Ga+As_pdπ", "Ga+As_psσ", "Ga+As_ppσ", "Ga+As_pdσ", "Ga+As_dpπ", "Ga+As_ddπ", "Ga+As_ddδ", "Ga+As_dsσ", "Ga+As_dpσ", "Ga+As_ddσ", "As+As_ssσ", "As+As_spσ", "As+As_sdσ", "As+As_ppπ", "As+As_pdπ", "As+As_ppσ", "As+As_pdσ", "As+As_ddπ", "As+As_ddδ", "As+As_ddσ"])
end

@testset "CsPbBr3 overlaps" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "s")
    set_value!(conf, "orbitals", "Pb", "s px py pz")
    set_value!(conf, "orbitals", "Br", "px py pz")
    set_value!(conf, "interpolate_rllm", false)
    
    strc_cspbbr3 = Structure(conf, poscar_path=cspbbr_poscar)
    orbitals_cspbbr3 = Hamster.get_orbitals(strc_cspbbr3, conf)

    overlaps_cspbbr3 = Hamster.get_overlaps(strc_cspbbr3.ions, orbitals_cspbbr3, conf)
    @test length(overlaps_cspbbr3) == 13
    @test sort(string.(overlaps_cspbbr3, apply_oc=true)) == sort(["Pb+Pb_ssσ", "Pb+Pb_spσ", "Pb+Pb_ppπ", "Pb+Pb_ppσ", "Cs+Pb_ssσ", "Cs+Pb_spσ", "Br+Pb_psσ", "Br+Pb_ppπ", "Br+Pb_ppσ", "Cs+Cs_ssσ", "Br+Cs_psσ", "Br+Br_ppπ", "Br+Br_ppσ"])
end

@testset "Matrix element" begin
    @test Hamster.me_to_overlap_label(Hamster.Vssσ((Hamster.s(), Hamster.s()))) == [0, 0, 0]
    @test Hamster.me_to_overlap_label(Hamster.Vspσ((Hamster.s(), Hamster.pz()))) == [0, 1, 0]
    @test Hamster.me_to_overlap_label(Hamster.Vppσ((Hamster.pz(), Hamster.pz()))) == [1, 1, 0]
    @test Hamster.me_to_overlap_label(Hamster.Vppπ((Hamster.px(), Hamster.px()))) == [1, 1, 1]
    @test Hamster.me_to_overlap_label(Hamster.Vppπ((Hamster.py(), Hamster.py()))) == [1, 1, 1]
    @test Hamster.me_to_overlap_label(Hamster.Vsdσ((Hamster.s(), Hamster.dz2()))) == [0, 2, 0]
end