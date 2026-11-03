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
@testset "POSCAR formats" begin
    reference = Hamster.read_poscar(test_file_path*"POSCAR_gaas")
    cartesian = reference.lattice * reference.rs_atom
    volume = abs(det(reference.lattice))
    variants = [
        "Selective dynamics" => """GaAs
            1.0
            2.825 2.825 0.0
            0.0 2.825 2.825
            2.825 0.0 2.825
            Ga As
            1 1
            Selective dynamics
            Direct
            0.0 0.0 0.0 T T F
            0.25 0.25 0.25 F F F
            """,
        "Cartesian, negative scale, POTCAR names" => """GaAs
            -$(volume)
            1.0 1.0 0.0
            0.0 1.0 1.0
            1.0 0.0 1.0
            Ga_d As/abc123
            1 1
            cartesian
            $(join(cartesian[:, 1] ./ 2.825, ' '))
            $(join(cartesian[:, 2] ./ 2.825, ' '))
            """]
    mktempdir() do dir
        for (name, content) in variants
            file = joinpath(dir, "POSCAR")
            write(file, content)
            poscar = Hamster.read_poscar(file)
            @test poscar.lattice ≈ reference.lattice
            @test poscar.rs_atom ≈ reference.rs_atom atol=1e-12
            @test poscar.atom_names == ["Ga", "As"]
            @test poscar.atom_types == reference.atom_types
        end
        write(joinpath(dir, "POSCAR_vasp4"), "GaAs\n1.0\n1 0 0\n0 1 0\n0 0 1\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n")
        @test_throws ErrorException Hamster.read_poscar(joinpath(dir, "POSCAR_vasp4"))
        write(joinpath(dir, "POSCAR_unknown"), "X\n1.0\n1 0 0\n0 1 0\n0 0 1\nXx\n1\nDirect\n0 0 0\n")
        @test_throws ErrorException Hamster.read_poscar(joinpath(dir, "POSCAR_unknown"))
    end
end
