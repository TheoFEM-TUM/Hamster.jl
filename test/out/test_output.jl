@testset "Warnings in hamster.out" begin
    mktempdir() do dir
        file = joinpath(dir, "hamster.out")
        logger = Hamster.OutputLogger(Logging.NullLogger(), filename=file)
        Logging.with_logger(logger) do
            @info "not in the output"
            @warn "before the header"
        end
        @test !isfile(file)

        write(file, "header\n")
        Hamster.start_output_log!(logger)
        Logging.with_logger(logger) do
            @warn "first line\nsecond line" n=3
            @warn "first line\nsecond line" n=3
            @warn "already in the output" _group=:hamster_out
            @error "failed"
        end
        Hamster.finish_output_log(logger)
        lines = readlines(file)
        @test lines[1:2] == ["header", "Warning: before the header"]
        @test lines[3:5] == ["Warning: first line", "  second line", "  n = 3"]
        @test lines[6] == "Error: failed"
        @test lines[end] == "The run gave 3 warnings, see the lines starting with \"Warning:\" above."
        @test count(line -> occursin("first line", line), lines) == 1
        @test !any(line -> occursin("already in the output", line) || occursin("not in the output", line), lines)
    end

    @test Hamster.output_logger(1) === Logging.current_logger()
    @test Hamster.output_logger(0) isa Hamster.OutputLogger
    logger = Hamster.OutputLogger(Logging.NullLogger())
    Logging.with_logger(() -> @warn("kept"), logger)
    Logging.with_logger(logger) do
        @test Hamster.output_logger(0) === logger
    end
    @test logger.pending == ["Warning: kept"] && logger.nwarnings == 1

    @test_logs (:warn, "passed on") Logging.with_logger(() -> @warn("passed on"),
                                                       Hamster.OutputLogger(Logging.current_logger(), filename=tempname()))

    mktempdir() do dir
        cd(dir) do
            @test_throws ErrorException Hamster.abort_on_error(() -> error("wrong input"), comm)
            @test occursin("ERROR: wrong input", read("hamster.out", String))
        end
    end
end
