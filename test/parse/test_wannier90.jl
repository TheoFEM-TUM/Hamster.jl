@testset "Wannier90 parser" begin
    path = joinpath(@__DIR__, "test_files")
    
    # Test 1: test real = true (default)
    Hr, Rs, deg = Hamster.read_hrdat(joinpath(path, "wannier90_hr.dat"), real=true)
    Hr_true = read_from_file(joinpath(path, "Hr_real_true.dat"), type=ComplexF64)
    all_correct = map(eachindex(Hr)) do R
        Hr[R] ≈ Hr_true[:, :, R]
    end
    @test all(all_correct)
    @test Rs ≈ read_from_file(joinpath(path, "Rs_true.dat"))
    @test deg ≈ read_from_file(joinpath(path, "deg_true.dat"))

    # Test 2: test real=false
    Hr_complex, _, _ = Hamster.read_hrdat(joinpath(path, "wannier90_hr.dat"), real=false)
    Hr_complex_true = read_from_file(joinpath(path, "Hr_complex_true.dat"), type=ComplexF64)
    all_correct_2 = map(eachindex(Hr)) do R
        Hr_complex[R] ≈ Hr_complex_true[:, :, R]
    end
    @test all(all_correct_2)
end