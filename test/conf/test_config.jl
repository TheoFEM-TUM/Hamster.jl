conf = Hamster.get_empty_config()

@testset "Config" begin
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
    Hamster.set_value!("rcut", "7.0", conf)
    @test conf("rcut") == 7.0

    # Value in non-existing block should be default
    @test conf("lr", "Optimizer") == "default"

    # Create block and set value in block
    Hamster.set_value!("lr", "Optimizer", "0.1", conf)
    @test conf("lr", "Optimizer") == 0.1

    # Lookup multiple keys
    @test conf(["rcut", "interpolate_rllm"]) == [7.0, "default"]
end