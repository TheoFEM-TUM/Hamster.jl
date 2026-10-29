@testset "KPOINTS" begin
    mktempdir() do dir
        write_file(name, text) = (file = joinpath(dir, name); write(file, text); file)

        # Line mode: N points per segment including both end points, labels ignored
        file = write_file("KPOINTS_line", """
            k-path
            3
            Line-mode
            Reciprocal
            0.0 0.0 0.0 ! G
            0.5 0.0 0.5 ! X

            0.5 0.0 0.5 ! X
            0.5 0.25 0.75 ! W
            """)
        ks = Hamster.read_kpoints(file)
        @test size(ks) == (3, 6)
        @test ks[:, 1:3] ≈ [0 0.25 0.5; 0 0 0; 0 0.25 0.5]
        @test ks[:, 4:6] ≈ [0.5 0.5 0.5; 0 0.125 0.25; 0.5 0.625 0.75]

        # Explicit list (weights ignored)
        file = write_file("KPOINTS_explicit", "explicit\n2\nReciprocal\n0 0 0 1\n0.5 0.5 0 2\n")
        @test Hamster.read_kpoints(file) ≈ [0 0.5; 0 0.5; 0 0]

        # Gamma-centered and Monkhorst-Pack grids
        ks = Hamster.read_kpoints(write_file("KPOINTS_gamma", "grid\n0\nGamma\n2 2 1\n"))
        @test size(ks) == (3, 4) && ks[:, 1] == zeros(3) && sort(unique(ks[1, :])) ≈ [0, 0.5] && all(ks[3, :] .== 0)
        ks = Hamster.read_kpoints(write_file("KPOINTS_mp", "grid\n0\nMonkhorst-Pack\n2 2 1\n0 0 0\n"))
        @test size(ks) == (3, 4) && sort(unique(ks[1, :])) ≈ [-0.25, 0.25] && all(ks[3, :] .== 0)

        # Cartesian coordinates depend on the lattice
        @test_throws ErrorException Hamster.read_kpoints(write_file("KPOINTS_cart", "explicit\n1\nCartesian\n0 0 0 1\n"))

        # The kpoints tag: a KPOINTS file gives the same k-points for every system
        conf = get_empty_config()
        set_value!(conf, "kpoints", joinpath(dir, "KPOINTS_line"))
        @test Hamster.get_kpoints_from_config(conf, system="a") == Hamster.get_kpoints_from_config(conf, system="b") ==
              Hamster.read_kpoints(joinpath(dir, "KPOINTS_line"))
        set_value!(conf, "kpoints", joinpath(dir, "unknown"))
        @test_throws ErrorException Hamster.get_kpoints_from_config(conf)
    end
end
