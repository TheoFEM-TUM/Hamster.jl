conf_path = string(@__DIR__)*"/hconf"
conf = get_config(filename=conf_path)

@testset "Config Read" begin
    @test conf("run_def") == true
    @test conf("n", "Ga") == 3
    @test conf("orbitals", "Ga") == ["sp3dr2", "sp3dr2", "sp3dr2" ,"sp3dr2"]

    # test sp_geo=true without spaces
    @test conf("sp_geo") == true
end
@testset "Comments in the config file" begin
    @test Hamster.filter_comment("# begin Supercell") == ""
    @test Hamster.filter_comment("    # kpoints = EIGENVAL") == ""
    @test Hamster.filter_comment("rcut = 5.5  # comment") == "rcut = 5.5"
    @test Hamster.filter_comment("rcut = 5.5") == "rcut = 5.5"
    mktempdir() do dir
        file = joinpath(dir, "hconf")
        write(file, """
        # A model: from the beginning to the end
        begin Options    # options
            rcut = 5.5   # TB cutoff, used from the beginning to the end
        # kpoints = EIGENVAL
        end
        # begin Supercell
        #     Nconf = 5
        # end
        begin atom Pb
            alpha = 37   ! Fortran-style comment
        end
        """)
        conf = Hamster.read_config(file)
        @test conf("rcut") == 5.5
        @test conf("kpoints") == "default"
        @test !haskey(conf.blocks, "Supercell")
        @test conf("alpha", "Pb") == 37
    end
end
