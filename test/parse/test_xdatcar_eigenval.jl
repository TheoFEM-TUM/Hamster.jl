@testset "XDATCAR with a variable cell and Cartesian coordinates" begin
    header(a) = "GaAs\n1.0\n$a 0 0\n0 $a 0\n0 0 $a\nGa As\n1 1\n"
    content = header(5.0) * "Direct configuration= 1\n0 0 0\n0.25 0.25 0.25\n" *
              header(6.0) * "Direct configuration= 2\n0 0 0\n0.5 0.25 0.25\n" *
              header(6.0) * "Cartesian configuration= 3\n0 0 0\n1.5 3.0 1.5\n"
    mktempdir() do dir
        file = joinpath(dir, "XDATCAR")
        write(file, content)
        lattice, configs = Hamster.read_xdatcar(file)
        @test size(lattice) == (3, 3, 3)
        @test lattice[:, :, 1] == 5.0 * I(3) && lattice[:, :, 2] == 6.0 * I(3) && lattice[:, :, 3] == 6.0 * I(3)
        @test configs[:, 2, 1] ≈ [0.25, 0.25, 0.25]
        @test configs[:, 2, 2] ≈ [0.5, 0.25, 0.25]
        @test configs[:, 2, 3] ≈ [0.25, 0.5, 0.25]
        _, cart = Hamster.read_xdatcar(file, frac=false)
        @test cart[:, 2, 1] ≈ [1.25, 1.25, 1.25]
        @test cart[:, 2, 2] ≈ [3.0, 1.5, 1.5]
        @test cart[:, 2, 3] ≈ [1.5, 3.0, 1.5]

        write(file, header(5.0) * "Direct configuration= 1\n0 0 0\n0.25 0.25 0.25\nDirect configuration= 2\n0 0 0\n0.5 0.5 0.5\n")
        lattice, configs = Hamster.read_xdatcar(file)
        @test lattice == 5.0 * I(3)
        @test size(configs) == (3, 2, 2)

        write(file, header(5.0) * "Direct configuration= 1\n0 0 0\n0.25 0.25 0.25\nDirect configuration= 2\n0 0 0\n")
        lattice, configs = @test_logs (:warn, r"incomplete") Hamster.read_xdatcar(file)
        @test size(configs) == (3, 2, 1)
    end
end

@testset "EIGENVAL of spin-polarized calculations" begin
    lines = readlines(joinpath(@__DIR__, "..", "calc", "test_files", "EIGENVAL_gaas"))
    mktempdir() do dir
        file = joinpath(dir, "EIGENVAL")
        write(file, join(vcat("    2    2    1    2", lines[2:end]), "\n"))
        @test_throws ErrorException Hamster.read_eigenval(file)
    end
    @test size(Hamster.read_eigenval(joinpath(@__DIR__, "..", "calc", "test_files", "EIGENVAL_gaas"))[2]) == (48, 56)
end
