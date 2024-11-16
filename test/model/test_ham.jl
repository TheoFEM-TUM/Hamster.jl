file_path = string(@__DIR__) * "/test_files/"

@testset "Phase factor" begin
    # Test arrays
    R⃗ = rand(3)
    k⃗ = rand(3)
    result = Hamster.exp_2πi(R⃗, k⃗)
    expected = exp(2π * im * dot(k⃗, R⃗))
    @test isapprox(result, expected, rtol=1e-10)
    @test result isa ComplexF64

    # Test matrices
    Rs = rand(3, 5)
    ks = rand(3, 7)
    result = Hamster.exp_2πi(Rs, ks)
    @test size(result) == (5, 7)

    # Test matrix-vector
    Rs = rand(3)
    ks = rand(3, 7)
    result = Hamster.exp_2πi(Rs, ks)
    @test size(result) == (1, 7)

    Rs = rand(3, 5)
    ks = rand(3)
    result = Hamster.exp_2πi(Rs, ks)
    @test size(result) == (5,)
end

@testset "Hamiltonian" begin
    N = 32; NR = 25; Nk = 40
    Rs = rand(3, NR); ks = rand(3, Nk)

    Hr_dense = [rand(N, N) for R in axes(Rs, 2)]
    Hr_sp = [sprand(N, N, 0.01) for R in axes(Rs, 2)]

    # Test empty Hamiltonians
    Hk_empty = Hamster.get_empty_complex_hamiltonians(N, Nk)
    @test length(Hk_empty) == Nk
    @test size(Hk_empty[1]) == (N, N)
    @test Hk_empty[1] isa Matrix{ComplexF64}

    Hk_empty = Hamster.get_empty_complex_hamiltonians(N, Nk, Hamster.Sparse())
    @test length(Hk_empty) == Nk
    @test size(Hk_empty[1]) == (N, N)
    @test Hk_empty[1] isa SparseMatrixCSC{ComplexF64, Int64}

    # Test get_hamiltonian
    Hk = get_hamiltonian(Hr_sp, Rs, ks)
    # Sparse Hr should be converted to dense Hk for sp_mode=false
    @test Hk[1] isa Matrix{ComplexF64}

    Hk = get_hamiltonian(Hr_sp, Rs, ks, Hamster.Sparse())
    # Sparse Hr should lead to sparse Hk for sp_mode=true
    @test Hk[1] isa SparseMatrixCSC{ComplexF64, Int64}
    
    kpoints = read_from_file(file_path*"kpoints.dat")
    Es_correct = read_from_file(file_path*"Es_correct.dat")
    Hr, Rs = read_hr(file_path*"gaas_hr.dat", verbose=0)
    
    # Test dense mode
    Hk1 = get_hamiltonian(Hr, Rs, kpoints)
    Es_dense, vs_dense = diagonalize(Hk1)
    @test Es_dense ≈ Es_correct

    # Test sparse mode
    Hr_sp, _ = read_hr(file_path*"gaas_hr.dat", sp_mode=true, verbose=0)
    @test Hr_sp isa Vector{SparseMatrixCSC{Float64, Int64}}

    Hk2 = get_hamiltonian(Hr_sp, Rs, kpoints)
    @test Hk2 isa Vector{Matrix{ComplexF64}}
    Es2, vs2 = diagonalize(Hk2)
    @test Es2 ≈ Es_correct
    @test vs_dense ≈ vs2

    # Test Neig
    Hk_sp = get_hamiltonian(Hr_sp, Rs, kpoints, Hamster.Sparse())
    Es_sp, _ = diagonalize(Hk_sp, Neig=3, target=-15)
    @test size(Es_sp) == (3, 80)
    @test round.(Es_sp, digits=5) ≈ round.(Es_correct[1:3, :], digits=5)

    # Test target
    VBM = maximum(Es_dense[4, :]); CBM = minimum(Es_dense[5, :])
    VBM_sp, _ = diagonalize(Hk_sp, target=VBM, Neig=1)
    @test size(VBM_sp) == (1, 80)
    @test VBM_sp[1, 20] ≈ VBM

    CBM_sp, _ = diagonalize(Hk_sp, target=CBM, Neig=1)
    @test size(CBM_sp) == (1, 80)
    @test CBM_sp[1, 20] ≈ CBM

    # Test write function
    Hr, Rs = read_hr(file_path*"gaas_hr.dat", verbose=0)
    write_hr(Hr, Rs, filename=file_path*"test_hr", verbose=0)
    Hr_read, Rs_read = read_hr(file_path*"test_hr.dat", verbose=0)
    @test all(Hr .≈ Hr_read)
    @test Rs == Rs_read
    rm(file_path*"test_hr.dat")

    # Test spin basis
    H = [1 0; 0 2]
    H_soc1 = Hamster.apply_spin_basis(H)
    @test size(H_soc1) == (4, 4)
    @test diag(H_soc1) == [1, 1, 2, 2]
    H_soc2 = Hamster.apply_spin_basis(H, alternating_order=true)
    @test size(H_soc2) == (4, 4)
    @test diag(H_soc2) == [1, 2, 1, 2]

    H_sp = sparse(H)
    H_soc_sp = Hamster.apply_spin_basis(H_sp)
    @test size(H_soc_sp) == (4, 4)
    @test typeof(H_soc_sp) <: typeof(H_sp)
    @test nonzeros(H_soc_sp) == [1, 1, 2, 2]

    # Test eigenvalues with soc, unchanged without soc matrix
    kpoints = read_from_file(file_path*"kpoints.dat")
    Es_correct = read_from_file(file_path*"Es_correct.dat")
    Hr, Rs = read_hr(file_path*"gaas_hr.dat", verbose=0)    
    Hr = Hamster.apply_spin_basis.(Hr)
    Hk = get_hamiltonian(Hr, Rs, kpoints)
    Es, _ = diagonalize(Hk)
    @test size(Es, 1) == 2*size(Es_correct, 1)
    @test Es[1:2:16, :] ≈ Es_correct
    @test Es[2:2:16, :] ≈ Es_correct

    dHr = [rand(8, 8) for R in 1:25]
    dHr_1 = Hamster.gradient_apply_spin_basis.(dHr)
    @test all(H->size(H)==(4, 4), dHr_1)
    dHr_2 = Hamster.gradient_apply_spin_basis.(dHr, alternating_order=true)
    @test all(H->size(H)==(4, 4), dHr_2)

    # Test reshape dense eigenvectors
    vs = rand(ComplexF64, 3, 4, 5)
    result = Hamster.reshape_and_sparsify_eigenvectors(vs, Hamster.Dense())
    @test size(result) == (4, 5)
    @test all(result[i, j] == vs[:, i, j] for i in 1:4, j in 1:5)

    # Test reshape into sparse eigenvectors
    vs = rand(ComplexF64, 3, 4, 5)
    vs[1, 1, 1] = 1e-5
    result = Hamster.reshape_and_sparsify_eigenvectors(vs, Hamster.Sparse(), sp_tol=1e-4)
    @test size(result) == (4, 5)
    @test all(result[i, j] isa SparseVector{ComplexF64, Int64} for i in 1:4, j in 1:5)
    @test all(isapprox(result[i, j][:], vs[:, i, j], atol=1e-4) for i in 1:4, j in 1:5)
    @test result[1, 1][1] == 0
end