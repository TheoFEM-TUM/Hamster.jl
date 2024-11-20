test_file_path = string(@__DIR__) * "/test_files/"

@testset "Methods" begin

    # Test 1: Test nearest neighbors
    poscar = Hamster.read_poscar(test_file_path*"POSCAR_CsPbBr3")
    strc = Structure(poscar_path=test_file_path*"POSCAR_CsPbBr3")

    rs_ion = Hamster.get_ion_positions(strc.ions)
    Ts = Hamster.frac_to_cart(strc.Rs, strc.lattice)

    # Pb has 6 nearest neighbor Br atoms
    rNNs = get_nearest_neighbors(rs_ion[1], rs_ion, Ts, strc.point_grid, kNN=7)
    δPb_Br = norm(Hamster.frac_to_cart(poscar.rs_atom[:, 1] - poscar.rs_atom[:, 3], poscar.lattice))
    
    nn_is_br = Bool[]
    for rNN in rNNs[1:6]
        push!(nn_is_br, Hamster.normdiff(rNN, rs_ion[1]) == δPb_Br)
    end
    @test all(nn_is_br)
    @test norm(rNNs[7]) ≠ δPb_Br

    # Each Br has 2 nearest neighbor Pb atoms
    for i in 3:5
        rNNs = get_nearest_neighbors(rs_ion[i], rs_ion, Ts, strc.point_grid, kNN=3)

        nn_is_Pb = Bool[]
        for rNN in rNNs[1:2]
            push!(nn_is_Pb, Hamster.normdiff(rNN, rs_ion[i]) == δPb_Br)
        end
        @test all(nn_is_Pb)
        @test norm(rNNs[3]) ≠ δPb_Br
    end

    # Cs has 6 nearest neighbor Br atoms
    rNNs = get_nearest_neighbors(rs_ion[2], rs_ion, Ts, strc.point_grid, kNN=7)

    δCs_Br = norm(Hamster.frac_to_cart(poscar.rs_atom[:, 2] - poscar.rs_atom[:, 3], poscar.lattice))
    nn_is_br = Bool[]
    for rNN in rNNs[1:6]
        push!(nn_is_br, Hamster.normdiff(rNN, rs_ion[2]) == δCs_Br)
    end
    @test all(nn_is_br)
    @test norm(rNNs[7]) ≠ δCs_Br
end