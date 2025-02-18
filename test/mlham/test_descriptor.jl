@testset "fcut function" begin
    fcut = Hamster.fcut

    # Test case 1: r > rcut should return 0
    @test fcut(2.0, 1.0) == 0.0
    @test fcut(5.0, 3.0) == 0.0

    # Test case 2: r = 0 should return 1
    @test fcut(0.0, 1.0) ≈ 1.0 atol=1e-6

    # Test case 3: r = rcut should return 0.0
    @test fcut(1.0, 1.0) ≈ 0.0 atol=1e-6
    @test fcut(2.0, 2.0) ≈ 0.0 atol=1e-6

    # Test case 4: r approaching rcut smoothly decreases to 0.5
    @test fcut(0.5, 1.0) > fcut(0.9, 1.0) > fcut(1.0, 1.0)

    # Test case 5: rcut = 0 should return correct behavior (avoiding division by zero)
    @test fcut(0.0, 0.0) == 1.0  # If rcut is 0, only r=0 should return 1
    @test fcut(0.1, 0.0) == 0.0  # Any r > 0 should return 0

    # Test case 6: r = rcut/2 should return an intermediate value
    @test fcut(0.5, 1.0) ≈ 0.5 atol=1e-4  # cos(π/2) = 0 -> (0+1)/2 = 0.5

    # Test case 7: Large rcut should still work
    @test fcut(5.0, 10.0) > 0.0  # Should not be exactly zero
    @test fcut(10.0, 10.0) ≈ 0.0 atol=1e-6  # Edge case

    # Test case 8: Negative r should still return a valid value
    @test fcut(-1.0, 1.0) ≈ 0.0 atol=1e-6  # Cosine function is even, so negative r behaves like positive
end

@testset "GaAs descriptors" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    #set_value!(conf, "sp_mode", true)
    #set_value!(conf, "sp_tol", 1e-5)
    set_value!(conf, "verbosity", 0)

    strc = Structure(conf)
    basis = Basis(strc, conf)
    model = TBModel(strc, basis, conf)

    #Test 1: test environmental descriptor
    env = Hamster.get_environmental_descriptor(model.hs, model.V, strc, basis, conf)

    @test env isa Vector{Float64}
    @test length(env) == length(basis)

    @test std(env[1:4]) < 1e-10
    @test std(env[5:8]) < 1e-10

    # Test 2: test orbswap
    type1 = "Ga"; type2 = "As"; iorb = 1; jorb = 2
    @test Hamster.decide_orbswap(type1, type2, iorb, jorb) == false
    @test Hamster.decide_orbswap(type1, type2, jorb, iorb) == false

    type1 = "As"; type2 = "Ga"; iorb = 1; jorb = 2
    @test Hamster.decide_orbswap(type1, type2, iorb, jorb) == true
    @test Hamster.decide_orbswap(type1, type2, jorb, iorb) == true

    type1 = "As"; type2 = "As"; iorb = 1; jorb = 2
    @test Hamster.decide_orbswap(type1, type2, iorb, jorb) == false
    @test Hamster.decide_orbswap(type1, type2, iorb, iorb) == false

    type1 = "As"; type2 = "As"; iorb = 2; jorb = 1
    @test Hamster.decide_orbswap(type1, type2, iorb, jorb) == true

    # Test 2: test angular descriptor
    # Test case 1: Identical atom types, no orbital swap
    itype = jtype = "As"
    ri = [0.0, 0.0, 0.0]
    rj = [1.0, 0.0, 0.0]
    iaxis = [0.0, 1.0, 0.0]
    jaxis = [0.0, 0.0, 1.0]
    orbswap = false

    φ, θs = Hamster.get_angular_descriptors(itype, jtype, ri, rj, iaxis, jaxis, orbswap)
    @test isapprox(φ, π/2, atol=1e-6)  # iaxis and jaxis are perpendicular
    @test θs == sort(θs)  # Must be sorted for same atom types

    itype, jtype = 1, 2
    orbswap = false
    φ, θs = Hamster.get_angular_descriptors(itype, jtype, ri, rj, iaxis, jaxis, orbswap)
    @test isapprox(φ, π/2, atol=1e-6)
    @test θs == [Hamster.calc_angle(iaxis, normalize(rj - ri)), Hamster.calc_angle(jaxis, normalize(ri - rj))]  # No sorting

    orbswap = true
    φ, θs = Hamster.get_angular_descriptors(itype, jtype, ri, rj, iaxis, jaxis, orbswap)
    @test isapprox(φ, π/2, atol=1e-6)
    @test θs == reverse([Hamster.calc_angle(iaxis, normalize(rj - ri)), Hamster.calc_angle(jaxis, normalize(ri - rj))])  # Reversed

    rj = ri
    φ, θs = Hamster.get_angular_descriptors(itype, jtype, ri, rj, iaxis, jaxis, orbswap)
    @test isapprox(φ, π/2, atol=1e-6)
    @test θs == reverse([Hamster.calc_angle(iaxis, normalize(iaxis)), Hamster.calc_angle(jaxis, normalize(jaxis))])

    iaxis = [1.0, 0.0, 0.0]
    jaxis = [1.0, 0.0, 0.0]  # Parallel axes
    φ, θs = Hamster.get_angular_descriptors(itype, jtype, ri, rj, iaxis, jaxis, orbswap)
    @test isapprox(φ, 0.0, atol=1e-6)  # Parallel axes should have angle 0

    # Test 3: test complete descriptor set
    descriptors = Hamster.get_tb_descriptor(model.hs, model.V, strc, basis, conf)
    @test descriptors isa Vector{SparseMatrixCSC{StaticArray{Tuple{8}, Float64, 1}, Int64}}

    # Test: test descriptor sampler
    X = rand(3, 1000)
    num_cluster = 10; num_points = 100
    Xout = Hamster.sample_structure_descriptors(X, Ncluster=num_cluster, Npoints=num_points)
    @test length(Xout) == num_points
    @test length(Xout[1]) == 3
end