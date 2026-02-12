@testset "Hamiltonian I/O" begin
    H_original = [rand(5,5), sprand(5,5,0.3)]  # mix of dense and sparse
    vecs_original = rand(3, 2)
    
    write_ham(H_original, vecs_original, comm, rank; space="k")
    H_loaded, vecs_loaded = read_ham(comm, rank; space="k")

    H_loaded_2, vecs_loaded_2 = read_ham(rank; space="k")
    
    @test eltype(typeof(vecs_loaded)) == Float64
    @test vecs_loaded ≈ vecs_original
    @test H_loaded == H_loaded_2
    @test vecs_loaded == vecs_loaded_2
    
    # Test that blocks are sparse and same shape
    @test all(issparse.(H_loaded))
    @test length(H_loaded) == length(H_original)
    for (h_orig, h_loaded) in zip(H_original, H_loaded)
        @test size(h_orig) == size(h_loaded)
        # Compare values (convert original to sparse if needed)
        h_sparse = issparse(h_orig) ? h_orig : sparse(h_orig)
        @test h_sparse ≈ h_loaded
    end

    rm("ham.h5", force=true)

    # Test for more R - vectors
    NR = 12
    Hrs = [rand(5, 5) for R in 1:NR]
    Rs = rand(-1:1, 3, NR)
    write_ham(Hrs, Rs, comm, rank; space="r")
    Hrs_loaded, Rs_loaded = read_ham(comm, rank; space="r")
    @test Hrs == Hrs_loaded
    @test Rs == Rs_loaded
    @test eltype(typeof(Rs_loaded)) == Int64

    rm("ham.h5", force=true)
end

@testset "Current I/O" begin
    ħ_eVfs = 0.6582119569
    filepath = joinpath(@__DIR__, "test_files")
    conf = get_config(filename=joinpath(filepath, "hconf_gaas"))
    set_value!(conf, "poscar", joinpath(filepath, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(filepath, "rllm_true.dat"))
    set_value!(conf, "verbosity", 0)
    
    strc = Structure(conf)
    basis = Basis(strc, conf)
    model = TBModel(strc, basis, conf)

    Hr = get_hr(model, Hamster.Dense())
    write_ham(Hr, strc.Rs, comm, rank; space="r")

    bonds = Hamster.get_bonds(strc, basis, conf)
    current_true = [-1im/ħ_eVfs .* bonds[R] .* Hr[R] for R in eachindex(Hr)]

    Hamster.write_current(bonds, comm, 0; filename="ham.h5", system="", rank=0, nranks=1)
    cx, cy, cz, vecs = Hamster.read_current(rank)

    @test vecs == strc.Rs
    @test eltype(typeof(vecs)) == Int64

    correct_current = Bool[]
    for R in eachindex(current_true), j in axes(current_true[R], 2), i in axes(current_true[R], 1)
        push!(correct_current, abs(current_true[R][i, j][1] - cx[R][i, j]) ≈ 0)
        push!(correct_current, abs(current_true[R][i, j][2] - cy[R][i, j]) ≈ 0)
        push!(correct_current, abs(current_true[R][i, j][3] - cz[R][i, j]) ≈ 0)
    end
    @test all(correct_current)

    rm("ham.h5", force=true)
end