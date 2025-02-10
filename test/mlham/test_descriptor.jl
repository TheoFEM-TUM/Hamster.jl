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

    # Test 3: test complete descriptor set
    descriptors = Hamster.get_tb_descriptor(model.hs, model.V, strc, basis, conf)

    # Test: test descriptor sampler
    X = rand(3, 1000)
    num_cluster = 10; num_points = 100
    Xout = Hamster.sample_structure_descriptors(X, Ncluster=num_cluster, Npoints=num_points)
    @test length(Xout) == num_points
    @test length(Xout[1]) == 3
end