gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"
cspbbr_poscar = string(@__DIR__) * "/../strc/test_files/POSCAR_CsPbBr3"


@testset "fcut function" begin
    fcut = Hamster.fcut

    # Test 1: r > rcut should return 0
    @test fcut(2.0, 1.0) == 0.0
    @test fcut(5.0, 3.0) == 0.0

    # Test 2: r = 0 should return 1
    @test fcut(0.0, 1.0) ≈ 1.0 atol=1e-6

    # Test 3: r = rcut should return 0.0
    @test fcut(1.0, 1.0) ≈ 0.0 atol=1e-6
    @test fcut(2.0, 2.0) ≈ 0.0 atol=1e-6

    # Test 4: r approaching rcut smoothly decreases to 0.5
    @test fcut(0.5, 1.0) > fcut(0.9, 1.0) > fcut(1.0, 1.0)

    # Test 5: rcut = 0 should return correct behavior (avoiding division by zero)
    @test fcut(0.0, 0.0) == 1.0  # If rcut is 0, only r=0 should return 1
    @test fcut(0.1, 0.0) == 0.0  # Any r > 0 should return 0

    # Test 6: r = rcut/2 should return an intermediate value
    @test fcut(0.5, 1.0) ≈ 0.5 atol=1e-4  # cos(π/2) = 0 -> (0+1)/2 = 0.5

    # Test 7: Large rcut should still work
    @test fcut(5.0, 10.0) > 0.0  # Should not be exactly zero
    @test fcut(10.0, 10.0) ≈ 0.0 atol=1e-6  # Edge case

    # Test 8: Negative r should still return a valid value
    @test fcut(-1.0, 1.0) ≈ 0.0 atol=1e-6  # Cosine function is even, so negative r behaves like positive

    # Test 9: Positive tolerance, r > rcut
    @test isapprox(fcut(5.5, 5.0, 1.0), fcut(0.5, 1.0))

    # Positive tolerance, r <= rcut
    @test fcut(4.0, 5.0, 1.0) == 1.0

    # Negative tolerance, inside range
    @test isapprox(fcut(9.5, 10.0, -1.0), fcut(0.5, 1.0))

    # Negative tolerance, outside range
    @test fcut(8.0, 10.0, -1.0) == 1.0

    # Edge case: exactly rcut
    @test isapprox(fcut(5.0, 5.0, 1.0), 1.0)
end

@testset "Rllm GaAs" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "interpolate_rllm", true)
    set_value!(conf, "rllm_file", string(@__DIR__)*"/test_files/rllm.dat")
    set_value!(conf, "verbosity", 0)

    strc_gaas = Structure(conf, poscar_path=gaas_poscar)
    orbitals_gaas = Hamster.get_orbitals(strc_gaas, conf)

    overlaps_gaas = Hamster.get_overlaps(strc_gaas.ions, orbitals_gaas, conf)

    Hamster.Random.seed!(42)
    rllm = Hamster.get_rllm(overlaps_gaas, conf)
    rllm_correct = Hamster.read_rllm(overlaps_gaas, comm, filename=string(@__DIR__)*"/test_files/rllm_true.dat")
    xs = range(0.0, 5.0, length=200)

    @test all(mean(abs.(rllm[key].(xs) .- rllm_correct[key].(xs))) < 2e-3 for key in keys(rllm))

    # Deterministic: independent of random numbers drawn before and of the order (and threads) of the interpolations
    Hamster.Random.seed!(1)
    rllm_reverse = Hamster.get_rllm(reverse(overlaps_gaas), conf)
    @test keys(rllm_reverse) == keys(rllm) && all(rllm_reverse[key].(xs) == rllm[key].(xs) for key in keys(rllm))

    # Overlap of two 1s Slater orbitals with the same exponent: S(R) = exp(-αR) (1 + αR + (αR)²/3)
    ss = overlaps_gaas[findfirst(o -> string(o.type) == "ssσ", overlaps_gaas)]
    for α in (1.0, 2.5), R in (0.0, 0.7, 2.0, 4.5)
        @test Hamster.distance_dependence(ss.type, ss.orbconfig, R, [1, 1], [α, α]) ≈ exp(-α * R) * (1 + α * R + (α * R)^2 / 3) atol=1e-6
    end
    rm(string(@__DIR__)*"/test_files/rllm.dat")

    # Test with .h5 format
    Hamster.save_rllm(rllm, comm, filename="rllm.h5")
    rllm_2 = Hamster.read_rllm(overlaps_gaas, comm, filename="rllm.h5")
    @test all(mean(abs.(rllm_2[key].(xs) .- rllm_correct[key].(xs))) < 2e-3 for key in keys(rllm))
    rm("rllm.h5")
end
# With several MPI ranks, every rank interpolates a part of the overlaps into a temporary file, which rank 0 merges into
# rllm_file. This also has to work if rllm_file is a text file (it failed before: the temporary .dat files were opened
# as HDF5). Rank 0 of two ranks is emulated in a single process.
@testset "Rllm from several ranks" begin
    conf = get_empty_config()
    for (element, alpha) in (("Ga", 9), ("As", 13))
        set_value!(conf, "orbitals", element, "s px py pz"); set_value!(conf, "n", element, 3); set_value!(conf, "alpha", element, alpha)
    end
    set_value!(conf, "verbosity", 0)
    basis = Basis(Structure(conf, poscar_path=gaas_poscar), conf)
    label(overlap) = string(overlap, apply_oc=true)
    labels_in(file) = endswith(file, ".h5") ? h5open(keys, file) : collect(keys(Hamster.read_rllm(basis.overlaps[1:0], comm, filename=file)))
    xs = range(0.0, 5.0, length=50)

    for rllm_file in ("rllm.dat", "rllm.h5")
        mktempdir() do dir
            cd(dir) do
                mkdir("tmp")
                Hamster.precalc_rllm([basis], conf, comm=comm, rank=0, nranks=2, rllm_file=rllm_file)
                @test isempty(readdir("tmp"))
                labels_rank0 = labels_in(rllm_file)
                @test 0 < length(labels_rank0) < length(unique(label.(basis.overlaps)))
                rllm_rank0 = Hamster.read_rllm(filter(o -> label(o) in labels_rank0, basis.overlaps), comm, filename=rllm_file)

                # Part of rank 1, with one overlap that is already in the file
                rank1 = filter(o -> !(label(o) in labels_rank0) || label(o) == labels_rank0[1], basis.overlaps)
                rllm_rank1 = Hamster.interpolate_overlaps!(Dict{String, Hamster.CubicSpline{Float64}}(), rank1, conf, verbosity=0)
                rllm_rank1[labels_rank0[1]] = Hamster.CubicSpline(collect(xs), zeros(length(xs)))
                Hamster.save_rllm(rllm_rank1, comm, filename=Hamster.rank_rllm_filename(rllm_file, 1))
                Hamster.combine_local_rllm_files(rllm_file, comm, rank=0, nranks=2)
                @test isempty(readdir("tmp"))

                rllm = Hamster.read_rllm(basis.overlaps, comm, filename=rllm_file)
                @test sort(labels_in(rllm_file)) == sort(unique(label.(basis.overlaps)))
                @test all(rllm[key].(xs) == rllm_rank0[key].(xs) for key in labels_rank0)
                @test all(rllm[key].(xs) == rllm_rank1[key].(xs) for key in keys(rllm_rank1) if !(key in labels_rank0))
            end
        end
    end
end
