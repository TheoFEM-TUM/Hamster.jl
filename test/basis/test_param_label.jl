@testset "IonLabel" begin
    @test Hamster.IonLabel(5, 3).types == [3, 5] # types are sorted
    @test Hamster.IonLabel(5, 3, sorted=false).types == [5, 3] # types are not sorted

    @test Hamster.IonLabel("As", "Ga").types == [31, 33]
    @test Hamster.IonLabel("As", "Ga") == Hamster.IonLabel("Ga", "As")
    @test Hamster.IonLabel("As", "Ga", sorted=false) ≠ Hamster.IonLabel("Ga", "As", sorted=false)

    ion_label1 = Hamster.IonLabel("Pb", "Br", sorted=false)
    ion_label2 = Hamster.IonLabel("Br", "Pb", sorted=false)
    @test isequal(ion_label1, ion_label2, sorted=true)
    @test !isequal(ion_label1, ion_label2, sorted=false)

    @test string(ion_label1) == "Pb+Br"
    @test string(ion_label2) == "Br+Pb"

    @test Hamster.aresameions(Hamster.IonLabel("Si", "Si"))

    @test Hamster.areswapped("Ga", "As") == false
    @test Hamster.areswapped("Ge", "Si") == true
    @test Hamster.areswapped(1, 2) == false
    @test Hamster.areswapped(5, 2) == true
end

@testset "NN Label" begin
    conf = get_empty_config()

    # Test 1: onsite=true, sepNN=false
    @test Hamster.get_nn_label(0., 1.5, conf) == 0
    @test Hamster.get_nn_label(0.1, 1.5, conf) == 1
    @test Hamster.get_nn_label(1.8, 1.5, conf) == 1

    # Test 2: onsite=false, sepNN=false
    set_value!(conf, "onsite", "false")
    @test Hamster.get_nn_label(0., 1.5, conf) == 1
    @test Hamster.get_nn_label(0.5, 1.5, conf) == 1
    @test Hamster.get_nn_label(1.6, 1.5, conf) == 1

    # Test 3: onsite=true, sepNN=true
    set_value!(conf, "onsite", "true")
    set_value!(conf, "sepNN", "true")
    @test Hamster.get_nn_label(1.5, 1.5, conf) == 1
    @test Hamster.get_nn_label(2.0, 1.5, conf) == 2
    @test Hamster.get_nn_label(0., 1.5, conf) == 0

    # Test 4: onsite=false, sepNN=true
    set_value!(conf, "onsite", "false")
    @test Hamster.get_nn_label(0.7, 1.5, conf) == 1
    @test Hamster.get_nn_label(1.5, 1.5, conf) == 1
    @test Hamster.get_nn_label(3.0, 1.5, conf) == 2
end

@testset "Overlap label" begin
    @test Hamster.string_to_overlap_label("ssσ") == [0, 0, 0]
    @test Hamster.string_to_overlap_label("sdσ") == [0, 2, 0]
    @test Hamster.string_to_overlap_label("ppπ") == [1, 1, 1]
    @test Hamster.string_to_overlap_label("ddδ") == [2, 2, 2]
    @test Hamster.string_to_overlap_label("diag1") == [1, 1, 0]
    @test Hamster.string_to_overlap_label("diag-2") == [-2, -2, 0]
    @test Hamster.string_to_overlap_label("offdiag-1-1") == [-1, -1, 1]
    @test Hamster.string_to_overlap_label("offdiag0-1") == [0, -1, 1]
    @test Hamster.string_to_overlap_label("offdiag-10") == [-1, -0, 1]
end

@testset "Param label" begin
    param_label1 = Hamster.ParamLabel(1, Hamster.IonLabel("Ga", "As"), SVector{3}([0, 0, 0]))
    @test param_label1 == Hamster.string_to_param_label("NN1_Ga+As_ssσ")
    @test string(param_label1) == "NN1_Ga+As_ssσ"
    @test Hamster.same_param_label(1, Hamster.IonLabel("Ga", "As"), SVector{3}([0, 0, 0]), param_label1)

    param_label2 = Hamster.ParamLabel(0, Hamster.IonLabel("Pb", "Pb"), SVector{3}([1, 1, 0]))
    @test param_label2 == Hamster.string_to_param_label("NN0_Pb+Pb_diag1")
    @test string(param_label2) == "NN0_Pb+Pb_diag1"
    @test Hamster.same_param_label(0, Hamster.IonLabel("Pb", "Pb"), SVector{3}([1, 1, 0]), param_label2)

    param_label3 = Hamster.ParamLabel(0, Hamster.IonLabel("Si", "Si"), SVector{3}([-2, -2, 1]))
    @test param_label3 == Hamster.string_to_param_label("NN0_Si+Si_offdiag-2-2")
    @test string(param_label3) == "NN0_Si+Si_offdiag-2-2"
    @test Hamster.same_param_label(0, Hamster.IonLabel("Si", "Si"), SVector{3}([-2, -2, 1]), param_label3)
end