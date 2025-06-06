@testset "Index maps" begin
    # Test 1: Test one ion with multiple orbitals
    @test Hamster.get_index_to_ion_orb_map([3]) == [(1, 1), (1, 2), (1, 3)]

    # Test 2: Test multiple ions with same number of orbitals
    @test Hamster.get_index_to_ion_orb_map([3, 3]) == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    #Test 3: Test multiple ions with different number of orbitals
    @test Hamster.get_index_to_ion_orb_map([3, 2]) == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

    # Test 4: Test multiple orbitals, some with 0 orbitals.
    @test Hamster.get_index_to_ion_orb_map([3, 0, 4]) == [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3), (3, 4)]
end