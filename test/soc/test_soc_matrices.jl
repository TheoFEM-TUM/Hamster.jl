@testset "soc matrices" begin
    # SOC matrices in atomic orbital basis are compared to Gu et al., Comput. Mater. Sci., 221, 112090 (2023)

    # Test 1: test s orbitals
    M_soc_s = Hamster.get_Msoc_s()
    @test M_soc_s ≈ zeros(2, 2)

    # Test 2: test p orbitals
    Msoc_p = 2 .* Hamster.get_Msoc_p(["pz↑", "px↑", "py↑", "pz↓", "px↓", "py↓"])

    @test Msoc_p ≈ ComplexF64[0 0 0 0 -1 im;
                              0 0 -im 1 0 0;
                              0 im 0 -im 0 0;
                              0 1 im 0 0 0;
                             -1 0 0 0 0 im;
                            -im 0 0 0 -im 0]

    # Test 3: test d orbitals
    Msoc_d = 2 .* Hamster.get_Msoc_d(["dz2↑", "dxz↑", "dyz↑", "dx2-y2↑", "dxy↑", "dz2↓", "dxz↓", "dyz↓", "dx2-y2↓", "dxy↓"])
    @test Msoc_d ≈ ComplexF64[0 0 0 0 0 0 -√3 im*√3 0 0;
                              0 0 -im 0 0 √3 0 0 -1 im;
                              0 im 0 0 0 -im*√3 0 0 -im -1;
                              0 0 0 0 -2*im 0 1 im 0 0;
                              0 0 0 2*im 0 0 -im 1 0 0;
                              0 √3 √3*im 0 0 0 0 0 0 0;
                            -√3 0 0 1 im 0 0 im 0 0;
                            -√3*im 0 0 -im 1 0 -im 0 0 0;
                              0 -1 im 0 0 0 0 0 0 2*im;
                              0 -im -1 0 0 0 0 0 -2*im 0]
end