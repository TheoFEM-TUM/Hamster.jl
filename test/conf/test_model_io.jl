using Hamster
using Test

@testset "Model IO" begin
    # Test 1: test model_file default
    conf = get_empty_config()
    @test Hamster.get_model_file(conf) == "model.h5"
    set_value!(conf, "model_file", "custom.h5")
    @test Hamster.get_model_file(conf) == "custom.h5"

    # Test 2: test load_model default
    conf = get_empty_config()
    @test Hamster.get_load_model(conf) == false
    set_value!(conf, "load_model", true)
    @test Hamster.get_load_model(conf) == true
    set_value!(conf, "load_model", "true")
    @test Hamster.get_load_model(conf) == true
    set_value!(conf, "load_model", "false")
    @test Hamster.get_load_model(conf) == false

    # Test 3: test save_model default
    conf = get_empty_config()
    @test Hamster.get_save_model(conf) == false
    set_value!(conf, "save_model", true)
    @test Hamster.get_save_model(conf) == true
    set_value!(conf, "save_model", "true")
    @test Hamster.get_save_model(conf) == true
    set_value!(conf, "save_model", "false")
    @test Hamster.get_save_model(conf) == false
end
