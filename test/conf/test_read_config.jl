conf_path = string(@__DIR__)*"/hconf"
conf = get_config(filename=conf_path)

@testset "Config Read" begin
    @test conf("run_def") == "true"
    @test conf("n", "Ga") == "3"
    @test conf("orbitals", "Ga") == "sp3dr2 sp3dr2 sp3dr2 sp3dr2"

    # test sp_geo=true without spaces
    @test conf("sp_geo") == "true"
end