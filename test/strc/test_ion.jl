@testset "Element/Number conversion" begin
    @test Hamster.element_to_number("Ga") == 31
    @test Hamster.element_to_number("Si") == 14
    @test Hamster.element_to_number("I") == 53


    @test Hamster.number_to_element(55) == "Cs"
    @test Hamster.number_to_element(115) == "Mc"
    @test Hamster.number_to_element(83) == "Bi"
end

@testset "Ion" begin
    # Test 1: Creation of an Ion instance
    ion = Ion("Na", SVector{3}(1.0, 2.0, 3.0), SVector{3}(0.1, 0.1, 0.1))
    
    @test ion.type == "Na"
    @test ion.pos == SVector{3}(1.0, 2.0, 3.0)
    @test ion.dist == SVector{3}(0.1, 0.1, 0.1)

    # Test 2: get_ions basic functionality
    positions = [1.0 4.0; 2.0 5.0; 3.0 6.0]
    types = ["Na", "Cl"]
    distortions = [0.1 0.2; 0.1 0.2; 0.1 0.2]

    ions = Hamster.get_ions(positions, types, distortions)

    @test length(ions) == 2
    @test ions[1].type == "Na"
    @test ions[1].pos == SVector{3}(1.0, 2.0, 3.0)
    @test ions[1].dist == SVector{3}(0.1, 0.1, 0.1)
    @test ions[2].type == "Cl"
    @test ions[2].pos == SVector{3}(4.0, 5.0, 6.0)
    @test ions[2].dist == SVector{3}(0.2, 0.2, 0.2)
    
    # Test 3: Empty inputs
    positions_empty = zeros(3, 0)
    types_empty = String[]
    distortions_empty = zeros(3, 0)

    ions_empty = Hamster.get_ions(positions_empty, types_empty, distortions_empty)

    @test length(ions_empty) == 0

    # Test 4: Single ion
    positions_single = [7.0; 8.0; 9.0]
    types_single = ["K"]
    distortions_single = [0.3; 0.3; 0.3]

    ions_single = Hamster.get_ions(positions_single, types_single, distortions_single)

    @test length(ions_single) == 1
    @test ions_single[1].type == "K"
    @test ions_single[1].pos == SVector{3}(7.0, 8.0, 9.0)
    @test ions_single[1].dist == SVector{3}(0.3, 0.3, 0.3)

    rs_ion = rand(3, 5)
    ion_types = ["H", "H", "O", "C", "Ge"]
    
    ions = Hamster.get_ions(rs_ion, ion_types)
    @test Hamster.get_ion_types(ions) == ion_types
    @test Hamster.get_ion_types(ions, uniq=true) == unique(ion_types)
    @test Hamster.get_ion_types(ions, sorted=true) == sort(ion_types)
    @test Hamster.get_ion_types(ions, uniq=true, sorted=true) == sort(unique(ion_types))
end