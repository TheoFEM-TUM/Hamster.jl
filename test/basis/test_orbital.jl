gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"
cspbbr_poscar = string(@__DIR__) * "/../strc/test_files/POSCAR_CsPbBr3"

@testset "GaAs Orbitals" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    strc_gaas = Structure(conf, poscar_path=gaas_poscar)
    
    # Test 1: Default hybrid orbital orientation
    default_hybrid_axes = normalize.([SVector{3}([1., 1., 1.]), SVector{3}([1., -1., -1.]), SVector{3}([-1., 1., -1.]), SVector{3}([-1., -1., 1.])])
    orbitals_1 = Hamster.get_orbitals(strc_gaas, conf)

    @test sum(Hamster.get_number_of_orbitals(orbitals_1)) == 8
    @test [orbital.axis for orbital in orbitals_1[1]] == default_hybrid_axes
    @test [orbital.axis for orbital in orbitals_1[2]] == default_hybrid_axes
    @test all([typeof(orbital.type) == Hamster.sp3dr2 for i in 1:2 for orbital in orbitals_1[i]])
    @test all([orbital.ion_type == 31 for orbital in orbitals_1[1]])
    @test all([orbital.ion_type == 33 for orbital in orbitals_1[2]])

    # Test 2: NNaxes=true
    set_value!(conf, "NNaxes", "Ga", "true")
    set_value!(conf, "NNaxes", "As", "true")
    orbitals_2 = Hamster.get_orbitals(strc_gaas, conf)
    default_hybrid_axes_normalized = normalize.(default_hybrid_axes)
    
    @test count([Hamster.normdiff(normalize(orbital.axis), axis) < 1e-10 for orbital in orbitals_2[1] for axis in default_hybrid_axes_normalized]) == 4
    @test count([Hamster.normdiff(normalize(orbital.axis), axis) < 1e-10 for orbital in orbitals_2[2] for axis in (-1).*default_hybrid_axes_normalized]) == 4
end

@testset "CsPbBr3 Orbitals" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "s")
    set_value!(conf, "orbitals", "Pb", "s px py pz")
    set_value!(conf, "orbitals", "Br", "px py pz")
    strc_cspbbr3 = Structure(conf, poscar_path=cspbbr_poscar)
    
    # Test 1: Default hybrid orbital orientation
    orbitals = Hamster.get_orbitals(strc_cspbbr3, conf)
    expected_types = [[Hamster.s, Hamster.px, Hamster.py, Hamster.pz], [Hamster.s], [Hamster.px, Hamster.py, Hamster.pz], [Hamster.px, Hamster.py, Hamster.pz], [Hamster.px, Hamster.py, Hamster.pz]]
    expected_axes = [[SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.])], 
                    [SVector{3}([0., 0., 1.])], 
                    [SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.])], 
                    [SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.])], 
                    [SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.]), SVector{3}([0., 0., 1.])]]
    expected_ion_types = [[82, 82, 82, 82], [55], [35, 35, 35], [35, 35, 35], [35, 35, 35]]

    @test sum(Hamster.get_number_of_orbitals(orbitals)) == 14
    @test all([typeof(orbitals[iion][jorb].type) == expected_types[iion][jorb] for iion in eachindex(orbitals) for jorb in eachindex(orbitals[iion])])
    @test all([orbitals[iion][jorb].axis == expected_axes[iion][jorb] for iion in eachindex(orbitals) for jorb in eachindex(orbitals[iion])])
    @test all([orbitals[iion][jorb].ion_type == expected_ion_types[iion][jorb] for iion in eachindex(orbitals) for jorb in eachindex(orbitals[iion])])

    orb_axes = Hamster.get_axes_from_orbitals(orbitals)
    @test all([orb_axes[iion][:, jorb] == expected_axes[iion][jorb] for iion in eachindex(orbitals) for jorb in eachindex(orbitals[iion])]) 
end