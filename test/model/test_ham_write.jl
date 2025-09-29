@testset "Hamiltonian I/O" begin
    H_original = [rand(5,5), sprand(5,5,0.3)]  # mix of dense and sparse
    vecs_original = rand(3, 5)
    
    write_ham(H_original, vecs_original, comm, rank; space="k")
    H_loaded, vecs_loaded = read_ham(comm, rank; space="k")

    H_loaded_2, vecs_loaded_2 = read_ham(rank; space="k")
    
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
end

if rank == 0
    rm("ham.h5", force=true)
end