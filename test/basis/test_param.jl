gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"
cspbbr_poscar = string(@__DIR__) * "/../strc/test_files/POSCAR_CsPbBr3"

@testset "Parameter label" begin
    param_label1 = Hamster.ParameterLabel(1, Hamster.IonLabel("Ga", "As"), SVector{3}([0, 0, 0]))
    @test param_label1 == Hamster.string_to_param_label("NN1_Ga+As_ssσ")
    @test string(param_label1) == "NN1_Ga+As_ssσ"
    @test Hamster.same_param_label(1, Hamster.IonLabel("Ga", "As"), SVector{3}([0, 0, 0]), param_label1)

    param_label2 = Hamster.ParameterLabel(0, Hamster.IonLabel("Pb", "Pb"), SVector{3}([1, 1, 0]))
    @test param_label2 == Hamster.string_to_param_label("NN0_Pb+Pb_diag1")
    @test string(param_label2) == "NN0_Pb+Pb_diag1"
    @test Hamster.same_param_label(0, Hamster.IonLabel("Pb", "Pb"), SVector{3}([1, 1, 0]), param_label2)

    param_label3 = Hamster.ParameterLabel(0, Hamster.IonLabel("Si", "Si"), SVector{3}([-2, -2, 1]))
    @test param_label3 == Hamster.string_to_param_label("NN0_Si+Si_offdiag-2-2")
    @test string(param_label3) == "NN0_Si+Si_offdiag-2-2"
    @test Hamster.same_param_label(0, Hamster.IonLabel("Si", "Si"), SVector{3}([-2, -2, 1]), param_label3)

    # Test: ParameterLabel as key in Dict
    ion_label1 = Hamster.IonLabel(1, 2)
    ion_label2 = Hamster.IonLabel(2, 3)

    p1 = ParameterLabel(1, ion_label1, @SVector [0, 0, 0])
    p2 = ParameterLabel(2, ion_label1, @SVector [1, 0, 0])
    p3 = ParameterLabel(1, ion_label2, @SVector [0, 1, 0])
    p4 = ParameterLabel(1, ion_label1, @SVector [0, 0, 0])  # identical to p1

    dict = Dict(p1 => "first", p2 => "second", p3 => "third")

    @test dict[p1] == "first"
    @test dict[p4] == "first"
    @test haskey(dict, p3)
    @test !haskey(dict, ParameterLabel(3, ion_label1, @SVector [0, 0, 0]))

    dict[p4] = "updated"
    @test dict[p1] == "updated"

    @test length(dict) == 3

    # Check that param label is a bits type (compatible with MPI operations)
    @test isbitstype(ParameterLabel)
end

@testset "Parameters GaAs" begin
    conf = get_empty_config()
    set_value!(conf, "system", "GaAs")
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "interpolate_rllm", false)
    
    strc_gaas = Structure(conf, poscar_path=gaas_poscar)
    orbitals_gaas = Hamster.get_orbitals(strc_gaas, conf)
    overlaps_gaas = Hamster.get_overlaps(strc_gaas.ions, orbitals_gaas, conf)

    parameters = Hamster.ParameterLabel[]
    
    # Test 1: Test onsite parameters
    Hamster.get_onsite_parameters!(parameters, overlaps_gaas)
    @test length(parameters) == 4
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Ga", "Ga"), SVector{3, Int64}([-2, -2, 0])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Ga", "Ga"), SVector{3, Int64}([-2, -2, 1])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("As", "As"), SVector{3, Int64}([-2, -2, 0])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("As", "As"), SVector{3, Int64}([-2, -2, 1])) ∈ parameters

    # Test 2: Test NN parameters
    Hamster.get_parameter_for_nn_label!(parameters, overlaps_gaas, 1)
    @test length(parameters) == 34
    parameters_c, Vs = read_params(string(@__DIR__)*"/test_files/gaas_params_nn1.dat")
    @test sort(string.(parameters)) == sort(string.(parameters_c))

    # Test 3: Test NN2 parameters
    Hamster.get_parameter_for_nn_label!(parameters, overlaps_gaas, 2)
    @test length(parameters) == 64
    parameters_c, Vs = read_params(string(@__DIR__)*"/test_files/gaas_params_nn2.dat")
    @test sort(string.(parameters)) == sort(string.(parameters_c))

    # Test 4: Test parameter read and write
    Vs = rand(length(parameters))
    write_params(parameters, Vs, conf, filename=string(@__DIR__)*"/params")

    parameters2, parameter_values2, _, _, conf_values = read_params(string(@__DIR__)*"/params.dat")
    @test parameters == parameters2
    @test Vs == parameter_values2
    @test Hamster.check_consistency(conf_values, conf)

    rm(string(@__DIR__)*"/params.dat")
end

@testset "Parameters GaAs overlaps (d-orbs)" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "s px py pz dxy dyz dxz dz2 dx2_y2")
    set_value!(conf, "orbitals", "As", "s px py pz dxy dyz dxz dz2 dx2_y2")
    set_value!(conf, "interpolate_rllm", false)
    
    strc_gaas = Structure(conf, poscar_path=gaas_poscar)
    orbitals_gaas = Hamster.get_orbitals(strc_gaas, conf)

    overlaps_gaas = Hamster.get_overlaps(strc_gaas.ions, orbitals_gaas, conf)
    params_gaas = Hamster.get_parameters_from_overlaps(overlaps_gaas, conf, onsite=true)
    @test length(overlaps_gaas) == 34
    @test length(params_gaas) == 10*3 + 8*2
end

@testset "Parameters CsPbBr3" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "s")
    set_value!(conf, "orbitals", "Pb", "s px py pz")
    set_value!(conf, "orbitals", "Br", "px py pz")
    set_value!(conf, "interpolate_rllm", false)
    
    strc_cspbbr3 = Structure(conf, poscar_path=cspbbr_poscar)
    orbitals_cspbbr3 = Hamster.get_orbitals(strc_cspbbr3, conf)
    overlaps_cspbbr3 = Hamster.get_overlaps(strc_cspbbr3.ions, orbitals_cspbbr3, conf)

    parameters = Hamster.ParameterLabel[]
    Hamster.get_onsite_parameters!(parameters, overlaps_cspbbr3)
    @test length(parameters) == 7
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Cs", "Cs"), SVector{3, Int64}([0, 0, 0])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Pb", "Pb"), SVector{3, Int64}([0, 0, 0])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Pb", "Pb"), SVector{3, Int64}([1, 1, 0])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Pb", "Pb"), SVector{3, Int64}([0, 1, 1])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Pb", "Pb"), SVector{3, Int64}([1, 1, 1])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Br", "Br"), SVector{3, Int64}([1, 1, 0])) ∈ parameters
    @test Hamster.ParameterLabel(0, Hamster.IonLabel("Br", "Br"), SVector{3, Int64}([1, 1, 1])) ∈ parameters

    # Test 2: Test NN parameters
    Hamster.get_parameter_for_nn_label!(parameters, overlaps_cspbbr3, 1)
    @test length(parameters) == 20
    parameters_c, Vs = read_params(string(@__DIR__)*"/test_files/cspbbr3_params.dat")
    @test sort(string.(parameters)) == sort(string.(parameters_c))
end