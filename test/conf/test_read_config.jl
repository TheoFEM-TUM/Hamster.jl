conf_path = string(@__DIR__)*"/hconf"
conf = get_config(filename=conf_path)

@testset "Config Read" begin
    @test conf("run_def") == true
    @test conf("n", "Ga") == 3
    @test conf("orbitals", "Ga") == ["sp3dr2", "sp3dr2", "sp3dr2" ,"sp3dr2"]

    # test sp_geo=true without spaces
    @test conf("sp_geo") == true
    @test conf("sp_tol") == 1e-5
    @test conf("log_modes", "HyperOpt") == ["log", "log"]

    # option values "begin"/"end" do not open or close blocks
    blocks = Hamster.split_blocks(["begin Optimizer", "onsite_shift = end", "train_mode = universal", "end",
                                   "begin Supercell", "XDATCAR = data.h5", "end"])
    @test [block.header for block in blocks] == [["Optimizer"], ["Supercell"]]
    @test blocks[1].content == ["onsite_shift = end", "train_mode = universal"]
end