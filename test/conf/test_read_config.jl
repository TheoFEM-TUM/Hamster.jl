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

@testset "Renamed input tags" begin
    mktempdir() do dir
        file = joinpath(dir, "hconf")
        write(file, """
        begin Options
            diag_method = chebyshev
        end
        """)
        conf = @test_logs (:warn, r"\"diag_method\" was renamed to \"eigsolver\"") Hamster.read_config(file)
        @test Hamster.get_eigsolver(conf) == "chebyshev"
        @test !haskey(conf.options, "diag_method")
    end
end

@testset "Errors for broken config files" begin
    mktempdir() do dir
        file = joinpath(dir, "hconf")
        read_text(text) = (write(file, text); Hamster.read_config(file))
        for (text, message) in (("begin Options\n  rcut 5\nend\n", r"line 2: \"rcut 5\" is not of the form tag = value"),
                                ("begin Options\n  rcut = 5 = 6\nend\n", r"line 2: .* is not of the form tag = value"),
                                ("begin Options\n  rcut =\nend\n", r"line 2: \"rcut =\" has no value"),
                                ("begin Options\n  = 5\nend\n", r"line 2: .* has no tag name"),
                                ("begin Options\n  rcut = 5\n", r"line 1: block without \"end\""),
                                ("begin Options\n  rcut = 5\nbegin ML\nend\n", r"line 3: \"begin\" inside the block of line 1"),
                                ("rcut = 5\nend\n", r"line 2: \"end\" without \"begin\""),
                                ("begin\n  rcut = 5\nend\n", r"line 1: \"begin\" without a block name"))
            @test_throws message read_text(text)
        end
        conf = read_text("notes outside the blocks\nbegin Options # options\n\n  # rcut = 6\n  rcut = 5 ! cutoff\nend\n")
        @test conf("rcut") == 5
    end
end

@testset "Values of input tags" begin
    conf = get_empty_config()
    set_value!(conf, "rcut", "5,5"); set_value!(conf, "sp_mode", "yes"); set_value!(conf, "Niter", "Optimizer", 2.5)
    set_value!(conf, "loss", "Optimizer", "RMSE"); set_value!(conf, "sampling", "ML", "kmeans"); set_value!(conf, "n", "Pb", "x")
    set_value!(conf, "wE", "Optimizer", "1 a"); set_value!(conf, "eigsolver", "shift_invert")
    messages = Hamster.check_config_values(conf)
    @test length(messages) == 8
    for pattern in (r"rcut \(Options block\): \"5,5\" is not a number", r"sp_mode .* is not true or false",
                    r"Niter \(Optimizer block\): \"2.5\" is not an integer"i, r"loss .* is not one of MAE, MSE",
                    r"sampling \(ML block\)", r"n \(atom Pb block\): \"x\" is not an integer", r"wE .* is not a list of numbers",
                    r"eigsolver .* is not one of shift-invert, krylov-schur, chebyshev")
        @test any(m -> occursin(pattern, m), messages)
    end
    @test_throws r"Invalid input tag values in my_config" Hamster.assert_valid_config(conf, source="my_config")

    valid = get_empty_config()
    set_value!(valid, "rcut", 5); set_value!(valid, "Niter", "Optimizer", "1e3"); set_value!(valid, "loss", "Optimizer", "mse")
    set_value!(valid, "mode", "ML", "Expand"); set_value!(valid, "sampling", "ML", "f"); set_value!(valid, "eigsolver", "Chebyshev")
    set_value!(valid, "orbitals", "Pb", "s px py pz"); set_value!(valid, "system", 2024); set_value!(valid, "method", "Ewald", "Wolf")
    @test isempty(Hamster.check_config_values(valid))
    @test Hamster.get_loss(valid) == "MSE" && Hamster.get_ml_mode(valid) == "expand" && Hamster.get_eigsolver(valid) == "chebyshev"
    @test Hamster.get_ewald_method(valid) == "wolf" && Hamster.get_system(valid) == "2024" && Hamster.get_niter(valid) == 1000
    @test occursin("Allowed values: MAE, MSE", sprint(Hamster.tag_help, "loss"))

    mktempdir() do dir
        file = joinpath(dir, "hconf")
        write(file, "begin Optimizer\n  loss = RMSE\nend\n")
        @test_throws r"loss \(Optimizer block\): \"RMSE\" is not one of MAE, MSE" Hamster.read_config(file)
    end
end
