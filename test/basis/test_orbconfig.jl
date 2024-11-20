@testset "Orbital configuration" begin
    # Test 1: Test conjugates
    sym = Hamster.SymOrb()
    def = Hamster.DefOrb()
    mirr = Hamster.MirrOrb()


    @test typeof(Hamster.conjugate(sym)) <: Hamster.SymOrb
    @test typeof(Hamster.conjugate(def)) <: Hamster.MirrOrb
    @test typeof(Hamster.conjugate(mirr)) <: Hamster.DefOrb

    # Test 2: s orbital on same ion type -> SymOrb
    orb_11 = Hamster.s(); orb_12 = Hamster.s()
    type_11 = 33; type_12 = 33
    Ys_11 = Hamster.get_orbital_list(Hamster.s()); Ys_12 = Hamster.get_orbital_list(Hamster.s())
    @test typeof(Hamster.OrbitalConfiguration(orb_11, orb_12, type_11, type_12, Ys_11, Ys_12)) <: Hamster.SymOrb

    # Test 3: s orbital on different ion type (no ion swap) -> SymOrb
    orb_21 = Hamster.s(); orb_22 = Hamster.s()
    type_21 = 33; type_22 = 34
    Ys_21 = Hamster.get_orbital_list(Hamster.s()); Ys_22 = Hamster.get_orbital_list(Hamster.s())
    @test typeof(Hamster.OrbitalConfiguration(orb_21, orb_22, type_21, type_22, Ys_21, Ys_22)) <: Hamster.SymOrb

    # Test 4: s orbital on different ion type (with ion swap) -> SymOrb
    orb_31 = Hamster.s(); orb_32 = Hamster.s()
    type_31 = 34; type_32 = 33
    Ys_31 = Hamster.get_orbital_list(Hamster.s()); Ys_32 = Hamster.get_orbital_list(Hamster.s())
    @test typeof(Hamster.OrbitalConfiguration(orb_31, orb_32, type_31, type_32, Ys_31, Ys_32)) <: Hamster.SymOrb

    # Test 5: s orbital on first, px orbital on second ion, different ion type (no ion swap) -> DefOrb
    orb_41 = Hamster.s(); orb_42 = Hamster.px()
    type_41 = 33; type_42 = 34
    Ys_41 = Hamster.get_orbital_list(Hamster.s()); Ys_42 = Hamster.get_orbital_list(Hamster.px())
    @test typeof(Hamster.OrbitalConfiguration(orb_41, orb_42, type_41, type_42, Ys_41, Ys_42)) <: Hamster.DefOrb

    # Test 6: s orbital on first, px orbital on second ion, different ion type (with ion swap) -> MirrOrb
    orb_51 = Hamster.s(); orb_52 = Hamster.px()
    type_51 = 34; type_52 = 33
    Ys_51 = Hamster.get_orbital_list(Hamster.s()); Ys_52 = Hamster.get_orbital_list(Hamster.px())
    @test typeof(Hamster.OrbitalConfiguration(orb_51, orb_52, type_51, type_52, Ys_51, Ys_52)) <: Hamster.MirrOrb

    # Test 7: px orbital on first, s orbital on second ion, different ion type (with ion and orb swap) -> DefOrb
    orb_61 = Hamster.px(); orb_62 = Hamster.s()
    type_61 = 34; type_62 = 33
    Ys_61 = Hamster.get_orbital_list(Hamster.px()); Ys_62 = Hamster.get_orbital_list(Hamster.s())
    @test typeof(Hamster.OrbitalConfiguration(orb_61, orb_62, type_61, type_62, Ys_61, Ys_62)) <: Hamster.DefOrb

    # Test 8: px orbital on first, s orbital on second ion, different ion type (with orb swap) -> MirrOrb
    orb_71 = Hamster.px(); orb_72 = Hamster.s()
    type_71 = 33; type_72 = 34
    Ys_71 = Hamster.get_orbital_list(Hamster.px()); Ys_72 = Hamster.get_orbital_list(Hamster.s())
    @test typeof(Hamster.OrbitalConfiguration(orb_71, orb_72, type_71, type_72, Ys_71, Ys_72)) <: Hamster.MirrOrb

    # Test 9: px orbital on first, s orbital on second, hybrid orbital, different ion (orb swap) -> MirrOrb
    orb_81 = Hamster.px(); orb_82 = Hamster.s()
    type_81 = 33; type_82 = 34
    Ys_81 = Hamster.get_orbital_list(Hamster.sp3()); Ys_82 = Hamster.get_orbital_list(Hamster.sp3())
    @test typeof(Hamster.OrbitalConfiguration(orb_81, orb_82, type_81, type_82, Ys_81, Ys_82)) <: Hamster.MirrOrb

    # Test 10: px orbital on first, s orbital on second, hybrid orbital, same ion (orb swap) -> SymOrb
    orb_91 = Hamster.px(); orb_92 = Hamster.s()
    type_91 = 33; type_92 = 33
    Ys_91 = Hamster.get_orbital_list(Hamster.sp3()); Ys_92 = Hamster.get_orbital_list(Hamster.sp3())
    @test typeof(Hamster.OrbitalConfiguration(orb_91, orb_92, type_91, type_92, Ys_91, Ys_92)) <: Hamster.SymOrb
end