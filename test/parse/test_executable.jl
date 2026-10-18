@testset "hamster executable" begin
    hamster = joinpath(pkgdir(Hamster), "hamster")
    if Sys.isunix() && isfile(hamster)
        version_output = read(`$hamster --version`, String)
        @test occursin("Hamster.jl $(pkgversion(Hamster))", version_output)
        @test occursin("path:  $(pkgdir(Hamster))", version_output)

        help_output = read(`$hamster --help`, String)
        @test occursin("Usage:", help_output)
        @test all(occursin(command, help_output) for command in ("--install", "--update", "--version", "--help"))
        @test read(`$hamster -h`, String) == help_output

        @test occursin("--no-pull", read(`$hamster --update --help`, String))
        @test occursin("--add_test_exec", read(`$hamster --install --help`, String))
    end
end
