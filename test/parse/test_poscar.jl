test_file_path = string(@__DIR__ ) * "/test_files/"
poscar = Hamster.read_poscar(test_file_path*"POSCAR_gaas")

@testset "Element/Number conversion" begin
    @test Hamster.element_to_number("C") isa UInt8
    @test Hamster.element_to_number("Ga") == 31
    @test Hamster.element_to_number("Si") == 14
    @test Hamster.element_to_number("I") == 53


    @test Hamster.number_to_element(55) == "Cs"
    @test Hamster.number_to_element(115) == "Mc"
    @test Hamster.number_to_element(83) == "Bi"
end

@testset "GaAs POSCAR Read" begin
    @test poscar.a == 5.65
    @test poscar.atom_names == ["Ga", "As"]
    @test poscar.lattice == [2.825 0.0 2.825; 2.825 2.825 0.0; 0.0 2.825 2.825]
    @test poscar.rs_atom == [0.0 0.25; 0.0 0.25; 0.0 0.25]
    @test poscar.atom_numbers == [1, 1]
    @test poscar.atom_types == UInt8[Hamster.element_to_number("Ga"), Hamster.element_to_number("As")]
end