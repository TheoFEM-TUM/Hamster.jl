test_file_path = string(@__DIR__) * "/test_files/"

@testset "Structure" begin

    # Test 1: Create Structure from POSCAR
    poscar = Hamster.read_poscar(test_file_path*"POSCAR_CsPbBr3")
    strc = Structure(poscar_path=test_file_path*"POSCAR_CsPbBr3")

    @test strc.lattice == poscar.lattice
    @test all([poscar.atom_types[iion] == strc.ions[iion].type for iion in axes(poscar.rs_atom, 2)])
    rs_atom = Hamster.frac_to_cart(poscar.rs_atom, poscar.lattice)
    @test all([rs_atom[:, iion] == strc.ions[iion].pos for iion in axes(poscar.rs_atom, 2)])
    @test all([[0., 0., 0.] == strc.ions[iion].dist for iion in axes(poscar.rs_atom, 2)])

    # Test 2: Create Structure from arrays

    # Test 3: Test behavior with Config
    conf = Hamster.get_empty_config()
    strc_1 = Structure(poscar_path=test_file_path*"POSCAR_CsPbBr3")
    strc_1.point_grid.grid_size == Hamster.get_grid_size(conf)

    set_value!(conf, "rcut", "5.0")
    strc_2 = Structure(conf, poscar_path=test_file_path*"POSCAR_CsPbBr3")
    strc_2.point_grid.grid_size == Hamster.get_rcut(conf)

    set_value!(conf, "grid_size", "7.0")
    strc_3 = Structure(conf, poscar_path=test_file_path*"POSCAR_CsPbBr3")
    strc_3.point_grid.grid_size == 7.0
end