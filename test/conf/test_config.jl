@testset "Config" begin
    conf = Hamster.get_empty_config()

    # Test convert value
    convert_value = Hamster.convert_value
    @test convert_value("7.0") == 7.0
    @test convert_value("1 2 3") == [1, 2, 3]
    @test convert_value("1.0 2 3") == Float64[1, 2, 3]
    @test convert_value("true") == true
    @test convert_value("false") == false
    @test convert_value("True") == true
    @test convert_value("FALSE") == false
    @test convert_value("1 1 1; 2 2 2") == [1 1 1; 2 2 2]
    @test convert_value("1 1 1; 2 2 2.0") == Float64[1 1 1; 2 2 2]

    # Parameter that is not set should return "default"
    @test conf("rcut") == "default"

    # Test setting parameter values
    set_value!(conf, "rcut", "7.0")
    @test conf("rcut") == 7.0

    # Value in non-existing block should be default
    @test conf("lr", "Optimizer") == "default"

    # Create block and set value in block
    set_value!(conf, "lr", "Optimizer", "0.1")
    @test conf("lr", "Optimizer") == 0.1

    # Lookup multiple keys
    @test conf(["rcut", "interpolate_rllm"]) == [7.0, "default"]

    # Keys should no be case sensitive
    set_value!(conf, "POSCAR", "POSCAR_gaas")
    @test conf("poscar") == "POSCAR_gaas"
    @test conf("POSCAR") == conf("poscar")

    set_value!(conf, "poScAr", "Supercell", "SC_POSCAR")
    @test conf("poscar", "Supercell") == "SC_POSCAR"
    @test conf("POSCAR", "Supercell") == "SC_POSCAR"
end

import Hamster.@configtag, Hamster.ConfigTag, Hamster.CONFIG_TAGS, Hamster.get_tag

@testset "Config tags" begin
    conf = get_empty_config()
    @configtag some_float Float64 1e-5 "This is a tag of type Float64."
    float_tag = Hamster.ConfigTag{Float64}("some_float", _ -> 1e-5,
        "This is a tag of type Float64.")
    @test get_tag(conf, float_tag) == 1e-5
    set_value!(conf, "some_float", 2.1)
    @test get_tag(conf, float_tag) == 2.1
    @test get_some_float(conf) == 2.1
    test_tag = CONFIG_TAGS[end]
    @test test_tag.name == float_tag.name
    @test test_tag.description == float_tag.description
    @test test_tag.block == float_tag.block

    vector_tag = ConfigTag{Vector{Int64}}("some_vector", conf->[1, 2],
        "This is a tag of type Vector{Int64}.")
    @configtag vector_tag Vector{Float64} [1.] "This is a tag of type Vector{Float64}."
    set_value!(conf, "vector_tag", 1e-5)

    @test get_vector_tag(conf) == [1e-5]
end