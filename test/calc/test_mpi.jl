"""
    run_with_two_ranks(case, dir; timeout=1800) -> (finished, exitcode, output)

Run `test_files/mpi_run.jl <case> <dir>` with two MPI ranks. A run that does not finish within `timeout` seconds is
killed (`finished = false`).
"""
function run_with_two_ranks(case, dir; timeout=1800)
    script = joinpath(@__DIR__, "test_files", "mpi_run.jl")
    logfile = joinpath(dir, "mpi_$case.log")
    exitcode, finished = open(logfile, "w") do io
        cmd = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) -t 1 $script $case $dir`
        process = run(pipeline(addenv(cmd, "OMP_NUM_THREADS" => "1"), stdout=io, stderr=io), wait=false)
        finished = timedwait(() -> process_exited(process), timeout, pollint=1.0) == :ok
        finished || kill(process)
        wait(process)
        (process.exitcode, finished)
    end
    return finished, exitcode, read(logfile, String)
end

@testset "MPI: an error on one rank stops all ranks" begin
    mktempdir() do dir
        finished, exitcode, output = run_with_two_ranks("abort", dir, timeout=600)
        @test finished
        @test exitcode ≠ 0
        @test occursin("ERROR on MPI rank 1", output)
        @test occursin("test error on rank 1", output)
    end
end

@testset "MPI: standard runs with two ranks" begin
    mktempdir() do dir
        mkdir(joinpath(dir, "md")); mkdir(joinpath(dir, "universal"))
        finished, exitcode, output = run_with_two_ranks("runs", dir)
        @test finished && exitcode == 0
        exitcode == 0 || println(output)

        cd(joinpath(dir, "md")) do
            inds = h5read("hamster_out.h5", "config_inds")
            @test length(inds) == 2
            @test size(read_from_file("Es.dat"), 3) == 2
            h5open("ham.h5") do file
                @test all(haskey(file, "H$(space)__$ind") for space in ("k", "r"), ind in inds)
                @test haskey(file, "basis")
            end
            @test !any(startswith("hamster_tmp_"), readdir())
        end

        cd(joinpath(dir, "universal")) do
            @test size(read_from_file("Es_cspbbr3.dat"), 1) == 14
            @test size(read_from_file("Es_gaas.dat"), 1) == 8
            h5open("ham.h5") do file
                @test haskey(file, "basis_cspbbr3") && haskey(file, "basis_gaas")
                @test haskey(file, "Hr_cspbbr3_1") && haskey(file, "Hr_gaas_1")
            end
        end
    end
end
