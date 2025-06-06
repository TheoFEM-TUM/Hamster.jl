test_file_path = joinpath(@__DIR__, "test_files")

@testset "Structure" begin
    # Test 1: Create Structure from POSCAR
    poscar = Hamster.read_poscar(joinpath(test_file_path, "POSCAR_CsPbBr3"))
    strc = Structure(poscar_path=joinpath(test_file_path, "POSCAR_CsPbBr3"))

    @test strc.lattice == poscar.lattice
    @test all([poscar.atom_types[iion] == strc.ions[iion].type for iion in axes(poscar.rs_atom, 2)])
    rs_atom = Hamster.frac_to_cart(poscar.rs_atom, poscar.lattice)
    @test all([rs_atom[:, iion] == strc.ions[iion].pos for iion in axes(poscar.rs_atom, 2)])
    @test all([[0., 0., 0.] == strc.ions[iion].dist for iion in axes(poscar.rs_atom, 2)])

    # Test 2: Create Structure from arrays

    # Test 3: Test behavior with Config
    conf = Hamster.get_empty_config()
    strc_1 = Structure(poscar_path=joinpath(test_file_path, "POSCAR_CsPbBr3"))
    strc_1.point_grid.grid_size == Hamster.get_grid_size(conf)

    set_value!(conf, "rcut", "5.0")
    strc_2 = Structure(conf, poscar_path=joinpath(test_file_path, "POSCAR_CsPbBr3"))
    strc_2.point_grid.grid_size == Hamster.get_rcut(conf)

    set_value!(conf, "grid_size", "7.0")
    strc_3 = Structure(conf, poscar_path=joinpath(test_file_path, "POSCAR_CsPbBr3"))
    strc_3.point_grid.grid_size == 7.0

    # Test 4: test translation vectors
    conf = get_empty_config()
    set_value!(conf, "rcut", 5)
    strc_4 = Structure(conf, poscar_path=joinpath(test_file_path, "SC_POSCAR_gaas"))
    rs_ion = Hamster.get_ion_positions(strc_4.ions)
    Rs = eachcol(strc_4.Rs)

    all_rs_included = Bool[]
    Ts = Hamster.get_translation_vectors(2)
    for t in eachcol(Ts), r1 in rs_ion, r2 in rs_ion
        Δr = Hamster.normdiff(r1, r2, Hamster.frac_to_cart(t, strc_4.lattice))
        if Δr ≤ 5
            push!(all_rs_included, t ∈ Rs)
        end
    end
    @test all(all_rs_included)

    # Test 5: test sep_NN
    conf = get_empty_config()
    set_value!(conf, "rcut", 5)
    set_value!(conf, "sepNN", true)
    strc_5 = Structure(conf, poscar_path=joinpath(test_file_path, "SC_POSCAR_gaas"))
    nn_dict = Hamster.get_nn_thresholds(strc_5.ions, Hamster.frac_to_cart(strc_5.Rs, strc_5.lattice), strc_5.point_grid, conf)
    @test nn_dict[Hamster.IonLabel("Ga", "As", sorted=false)] ≈ Hamster.normdiff(strc_5.ions[1].pos, strc_5.ions[5].pos)
    @test nn_dict[Hamster.IonLabel("As", "Ga", sorted=false)] ≈ Hamster.normdiff(strc_5.ions[1].pos, strc_5.ions[5].pos)
    @test nn_dict[Hamster.IonLabel("As", "As", sorted=false)] ≈ Hamster.normdiff(strc_5.ions[1].pos, strc_5.ions[2].pos)
    @test nn_dict[Hamster.IonLabel("Ga", "Ga", sorted=false)] ≈ Hamster.normdiff(strc_5.ions[5].pos, strc_5.ions[6].pos)
end
