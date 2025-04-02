@testset "lms basis" begin
    # Test 1: test lms basis from s,p and d
    s_basis = Hamster.get_lms_basis("s")
    p_basis = Hamster.get_lms_basis("p")
    d_basis = Hamster.get_lms_basis("d")
    @test s_basis == [[0, 0, -1], [0, 0, 1]]
    @test p_basis == [[1, -1, -1], [1, -1, 1], [1, 0, -1], [1, 0, 1], [1, 1, -1], [1, 1, 1]]
    @test d_basis == [[2, -2, -1], [2, -2, 1], [2, -1, -1], [2, -1, 1], [2, 0, -1], [2, 0, 1], [2, 1, -1], [2, 1, 1], [2, 2, -1], [2, 2, 1]]

    # Test 2: test lms soc matrix
    s_matrix = Hamster.get_matrix_lmbasis(s_basis)
    @test size(s_matrix) == (2, 2)
    @test all(s_matrix .== 0)

    p_matrix = Hamster.get_matrix_lmbasis(p_basis)
    @test size(p_matrix) == (6, 6)
    @test Hermitian(p_matrix) == p_matrix
    
    d_matrix = Hamster.get_matrix_lmbasis(d_basis)
    @test size(d_matrix) == (10, 10)
    @test Hermitian(d_matrix) == d_matrix

    # Test 3: test Msoc in real basis
    Msoc_p = Hamster.trans_lm_spatial("p", p_matrix)
end