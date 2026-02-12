@testset "greedy farthest-point sampling" begin
    ds = rand(3, 100)
    cluster_indices = 1:5:100
    num_to_take = 5

    selected = Hamster.farthest_point_sampling(ds, cluster_indices, num_to_take)
    @test length(unique(selected)) == num_to_take
    @test all([i ∈ cluster_indices for i in selected])
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
    env = Hamster.get_environmental_descriptor(model.hs, model.params, strc, basis, conf)

    @test env isa Vector{Float64}
    @test length(env) == length(basis)

    @test std(env[1:4]) < 1e-10
    @test std(env[5:8]) < 1e-10

    # Test 2: test orbswap
    # Different elements: should swap if element_to_number(itype) > element_to_number(jtype)
    @test Hamster.decide_orbswap("Pb", "Ga", 0, 0, 0, 0) == true   # Pb > Ga
    @test Hamster.decide_orbswap("Br", "Cs", 0, 0, 0, 0) == false  # Br < Cs

    # Same element, l_i > l_j: should swap
    @test Hamster.decide_orbswap("As", "As", 2, 0, 1, 0) == true   # d vs p
    @test Hamster.decide_orbswap("As", "As", 1, 0, 2, 0) == false

    # Same element, same l, m_i > m_j: should swap
    @test Hamster.decide_orbswap("Ga", "Ga", 1, 1, 1, 0) == true
    @test Hamster.decide_orbswap("Ga", "Ga", 1, 0, 1, 1) == false

    # Same element, same l and m: no swap
    @test Hamster.decide_orbswap("Cs", "Cs", 0, 0, 0, 0) == false

    # Different elements, same orbitals: test periodic table order
    @test Hamster.decide_orbswap("Pb", "Br", 1, 0, 1, 0) == true   # Pb > Br
    @test Hamster.decide_orbswap("As", "Pb", 1, 0, 1, 0) == false  # As < Pb

    # Edge case: same element, same l, m_i > m_j
    @test Hamster.decide_orbswap("Br", "Br", 2, 2, 2, 1) == true

    # Test 2: test angular descriptor
    # Test case 1: Identical atom types, no orbital swap
    itype = jtype = "As"
    ri = [0.0, 0.0, 0.0]
    rj = [1.0, 0.0, 0.0]
    iaxis = [0.0, 1.0, 0.0]
    jaxis = [0.0, 0.0, 1.0]
    orbswap = false

    φ, θs = Hamster.get_angular_descriptors(ri, rj, iaxis, jaxis)
    @test isapprox(φ, π/2, atol=1e-6)  # iaxis and jaxis are perpendicular
    @test θs == sort(θs)  # Must be sorted for same atom types

    itype, jtype = 1, 2
    orbswap = false
    φ, θs = Hamster.get_angular_descriptors(ri, rj, iaxis, jaxis)
    @test isapprox(φ, π/2, atol=1e-6)
    @test θs == [Hamster.calc_angle(iaxis, normalize(rj - ri)), Hamster.calc_angle(jaxis, normalize(ri - rj))]

    orbswap = true
    φ, θs = Hamster.get_angular_descriptors(ri, rj, iaxis, jaxis)
    @test isapprox(φ, π/2, atol=1e-6)
    @test θs == [Hamster.calc_angle(iaxis, normalize(rj - ri)), Hamster.calc_angle(jaxis, normalize(ri - rj))]

    rj = ri
    φ, θs = Hamster.get_angular_descriptors(ri, rj, iaxis, jaxis)
    @test isapprox(φ, π/2, atol=1e-6)
    @test θs == [Hamster.calc_angle(iaxis, normalize(iaxis)), Hamster.calc_angle(jaxis, normalize(jaxis))]

    iaxis = [1.0, 0.0, 0.0]
    jaxis = [1.0, 0.0, 0.0]  # Parallel axes
    φ, θs = Hamster.get_angular_descriptors(ri, rj, iaxis, jaxis)
    @test isapprox(φ, 0.0, atol=1e-6)  # Parallel axes should have angle 0

    # Test 3: test complete descriptor set
    descriptors = Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, conf)
    @test descriptors isa Vector{SparseMatrixCSC{SVector{20, Float64}, Int64}}

    # Test: test descriptor sampler
    X = rand(3, 1000)
    num_cluster = 10; num_points = 100
    Xout = Hamster.sample_structure_descriptors(X, Ncluster=num_cluster, Npoints=num_points)
    @test length(Xout) == num_points
    @test length(Xout[1]) == 3
end

@testset "CsPbBr3 descriptors (Orbital ordering)" begin
    path = joinpath(@__DIR__, "test_files/")
    conf = get_config(filename=joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_CsPbBr3"))
    #set_value!(conf, "init_params", "ML", joinpath(path, "ml_params.dat"))
    set_value!(conf, "init_params", "ML", "zeros")
    set_value!(conf, "verbosity", 0)

    strc_1 = Structure(conf); basis_1 = Basis(strc_1, conf); model_1 = TBModel(strc_1, basis_1, conf)
    descriptors_1 = Hamster.get_tb_descriptor(model_1.hs, model_1.params, strc_1, basis_1, conf)

    set_value!(conf, "orbitals", "Pb", "s py px pz"); set_value!(conf, "orbitals", "Br", "py px pz")
    strc_2 = Structure(conf); basis_2 = Basis(strc_2, conf); model_2 = TBModel(strc_2, basis_2, conf)
    descriptors_2 = Hamster.get_tb_descriptor(model_2.hs, model_2.params, strc_2, basis_2, conf)

    orbs_1 = ["Pb_s", "Pb_px", "Pb_py", "Pb_pz", "Cs_s", "Br-1_px", "Br-1_py", "Br-1_pz", "Br-2_px", "Br-2_py", "Br-2_pz", "Br-3_px", "Br-3_py", "Br-3_pz"]
    orbs_2 = ["Pb_s", "Pb_py", "Pb_px", "Pb_pz", "Cs_s", "Br-1_py", "Br-1_px", "Br-1_pz", "Br-2_py", "Br-2_px", "Br-2_pz", "Br-3_py", "Br-3_px", "Br-3_pz"]
    correct_permutation = Bool[]
    for R in eachindex(descriptors_1), i in axes(descriptors_1[R], 1), j in axes(descriptors_1[R], 2)
        i_2 = findfirst(orb -> orb == orbs_1[i], orbs_2)
        j_2 = findfirst(orb -> orb == orbs_1[j], orbs_2)
        err = sum(abs.(descriptors_1[R][i, j] .- descriptors_2[R][i_2, j_2]))
        push!(correct_permutation, err < 1e-5)
        if err > 1e-5
            println("--- $(orbs_1[i]), $(orbs_1[j]) ---")
            @show descriptors_1[R][i, j]
            @show descriptors_2[R][i_2, j_2]
        end
    end
    @test all(correct_permutation)
end