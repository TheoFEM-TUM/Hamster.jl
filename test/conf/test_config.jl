conf = Hamster.get_empty_config()

@testset "Config" begin
    # Parameter that is not set should return "default"
    @test conf("rcut") == "default"

    # Test setting parameter values
    Hamster.set_value!("rcut", "7.0", conf)
    @test conf("rcut") == "7.0"

    # Value in non-existing block should be default
    @test conf("lr", "Optimizer") == "default"

    # Create block and set value in block
    Hamster.set_value!("lr", "Optimizer", "0.1", conf)
    @test conf("lr", "Optimizer") == "0.1"

    # Lookup multiple keys
    @test conf(["rcut", "interpolate_rllm"]) == ["7.0", "default"]
end