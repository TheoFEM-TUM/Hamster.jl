conf_path = string(@__DIR__)*"/hconf"
conf = get_config(filename=conf_path)

@testset "Config Read" begin
    @test conf("skip_diag") == true
    @test conf("n", "Ga") == 3
    @test conf("orbitals", "Ga") == ["sp3dr2", "sp3dr2", "sp3dr2" ,"sp3dr2"]
    @test conf("xdatcar", "Supercell") == "SC_XDATCAR"                  # upper-case keys
    @test conf("we_1-4", "Optimizer") == 0.5

    # sp_mode=true without spaces
    @test conf("sp_mode") == true

    # The example contains only valid tags: reading it gives no warnings
    @test_logs get_config(filename=conf_path)
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

@testset "Tutorial config files" begin
    # Complete config files (examples and solutions) contain only valid tags; every exercise has the same lines as its
    # solution, with ??? in place of some values
    tutorials = joinpath(@__DIR__, "..", "..", "tutorials")
    tag_lines(file) = filter(!isempty, strip.(Hamster.filter_comment.(readlines(file))))
    files = [joinpath(root, f) for (root, _, fs) in walkdir(tutorials) for f in fs if startswith(f, "hconf")]
    @test length(files) ≥ 15
    for file in files
        if occursin("???", read(file, String))
            solution = joinpath(dirname(file), "solution", basename(file))
            @test isfile(solution)
            lines, solution_lines = tag_lines(file), tag_lines(solution)
            @test length(lines) == length(solution_lines)
            @test all(zip(lines, solution_lines)) do (l, s)
                occursin("???", l) || l == s || replace(l, "../data" => "../../data") == s
            end
        else
            @test_logs get_config(filename=file)
        end
    end
end
