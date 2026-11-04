@testset "Translation Vectors" begin
    conf = Hamster.get_empty_config()

    lattice = [2.825 0.0 2.825; 2.825 2.825 0.0; 0.0 2.825 2.825]
    # Test 1: Default Rmax for rcut=0
    Hamster.set_value!(conf, "rcut", "0.")
    @test Hamster.get_Rmax(lattice, conf) == Hamster.get_Rmax(conf)

    # Test 2: Rmax parameter if set and rcut=0
    Hamster.set_value!(conf, "Rmax", "3")
    @test Hamster.get_Rmax(lattice, conf) == Hamster.get_Rmax(conf)

    # Test 3: 2 for rcut=7
    Hamster.set_value!(conf, "rcut", "7")
    @test Hamster.get_Rmax(lattice, conf, positions=lattice * [0 0.25; 0 0.25; 0 0.25]) == 2

    # Test 4: Check that no Rmax=3 is smaller than rcut
    larger_than_rcut = Bool[]
    for i in -3:3, j in -3:3, k in -3:3
        if any(@. abs([i, j, k]) == 3)
            push!(larger_than_rcut, norm(Hamster.frac_to_cart([i, j, k], lattice)) > 7)
        end
    end
    @test all(larger_than_rcut)

    # Test 5: Test findR0
    Rs1 = [1 0 0; 0 1 0; 0 0 1]  # Zero vector is not present, should return 0
    Rs2 = [1 0 0; 0 0 0; 0 0 1]  # Zero vector is present at column 2
    Rs3 = [0 0 0; 0 0 0; 0 0 0]  # Zero vector present in all columns, should return 1

    @test findR0(Rs1) == 0
    @test findR0(Rs2) == 2
    @test findR0(Rs3) == 1

    # Test 6: Test get_translation_vectors with integers
    Rs1 = Hamster.get_translation_vectors(1)
    @test maximum(Rs1) == 1
    @test size(Rs1, 2) == (2*1 + 1)^3
    @test length(unique(eachcol(Rs1))) == size(Rs1, 2)
    @test findR0(Rs1) ≠ 0

    Rs3 = Hamster.get_translation_vectors(3)
    @test maximum(Rs3) == 3
    @test size(Rs3, 2) == (2*3 + 1)^3
    @test length(unique(eachcol(Rs3))) == size(Rs3, 2)
    @test findR0(Rs3) ≠ 0

    # Test 7: Test get_translation_vectors with rcut
    lattice = [2.825 0.0 2.825; 2.825 2.825 0.0; 0.0 2.825 2.825]
    rs_ion = [0 1.3675; 0 1.3675; 0 1.3675]
    rcut = 7
    Hamster.set_value!(conf, "rcut", string(rcut))
    Rs_cut = Hamster.get_translation_vectors(rs_ion, lattice, conf)
    Rmax = Int(maximum(abs.(Rs_cut)) + 1)
    Rs_nocut = Hamster.get_translation_vectors(Rmax)

    larger_than_rcut = Bool[]
    smaller_than_rcut = Bool[]
    for R⃗ in eachcol(Rs_nocut)
        if R⃗ ∉ eachcol(Rs_cut)
            # For no combination of r1, r2 and R⃗ should the norm be smaller than rcut
            for r1 in eachcol(rs_ion), r2 in eachcol(rs_ion)
                push!(larger_than_rcut, Hamster.normdiff(r1, r2, Hamster.frac_to_cart(R⃗, lattice)) > rcut)
            end
        elseif R⃗ ∈ eachcol(Rs_cut)
            # For at least one combination should the norm be smaller than rcut
            all_combinations_smaller_rcut = Bool[]
            for r1 in eachcol(rs_ion), r2 in eachcol(rs_ion)
                push!(all_combinations_smaller_rcut, Hamster.normdiff(r1, r2, Hamster.frac_to_cart(R⃗, lattice)) ≤ rcut)
            end
            push!(smaller_than_rcut, any(all_combinations_smaller_rcut))
        end
    end
    @test all(smaller_than_rcut)
    @test all(larger_than_rcut)

    # Test 8: For rcut=0, all translation vectors for the Rmax setting are returned
    @test Hamster.get_translation_vectors(rs_ion, lattice, Rmax=3, rcut=0) == Hamster.get_translation_vectors(3)
end
@testset "Translation vectors of thin and skewed cells" begin
    function translations_within(rs, lattice, rcut; M=12)
        Rs = Hamster.get_translation_vectors(M)
        found = Set{Vector{Float64}}()
        for R in axes(Rs, 2), i in axes(rs, 2), j in axes(rs, 2)
            norm(rs[:, j] + lattice * Rs[:, R] - rs[:, i]) ≤ rcut && push!(found, Rs[:, R])
        end
        return found
    end
    rcut = 7.0
    cells = [[1.5 0 0; 0 12 0; 0 0 12],
             [2.0 0 0; 0 2.0 0; 0 0 2.0],
             [6.0 5.7 5.7; 0 1.8 0.6; 0 0 1.7]]
    for lattice in cells
        rs = lattice * [0.1 0.6; 0.2 0.7; 0.3 0.95]
        Rs = Hamster.get_translation_vectors(rs, lattice, rcut=rcut)
        @test translations_within(rs, lattice, rcut) == Set(eachcol(Rs))
    end

    lattice = [2.825 0.0 2.825; 2.825 2.825 0.0; 0.0 2.825 2.825]
    @test Hamster.get_Rmax(lattice, rcut=7.0, positions=lattice * [0 0.25; 0 0.25; 0 0.25]) == 2
    @test Hamster.get_Rmax(5.65 .* Matrix(1.0I, 3, 3), rcut=7.0) == 2
end
