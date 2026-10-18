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
    rm(string(@__DIR__)*"/test_files/rllm.dat")

    # Test with .h5 format
    Hamster.save_rllm(rllm, comm, filename="rllm.h5")
    rllm_2 = Hamster.read_rllm(overlaps_gaas, comm, filename="rllm.h5")
    @test all(mean(abs.(rllm_2[key].(xs) .- rllm_correct[key].(xs))) < 2e-3 for key in keys(rllm))
    rm("rllm.h5")
end