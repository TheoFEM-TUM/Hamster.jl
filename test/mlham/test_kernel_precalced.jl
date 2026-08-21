@testset "HamiltonianKernelPrecalced" begin
	point = SVector{8, Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	second_point = SVector{8, Float64}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	descriptor = sparse([1, 1], [1, 2], [point, second_point], 2, 2)
    conf = get_empty_config()
    set_value!(conf, "verbosity", 0)
	base_kernel = HamiltonianKernel([2.0, 3.0], [point, second_point], 0.5, [[descriptor]], true)
	kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf)

	@test Hamster.get_params(kernel) == [2.0, 3.0]
	@test kernel.kp.datapoints == [point, second_point]
	@test kernel.kp.key_ranges == Dict((0,) => 1:2)
	@test kernel.sm.sim_params == 0.5

	expected = [2.0 * Hamster.exp_sim2(point, point, σ=0.5) + 3.0 * Hamster.exp_sim2(second_point, point, σ=0.5),
				2.0 * Hamster.exp_sim2(point, second_point, σ=0.5) + 3.0 * Hamster.exp_sim2(second_point, second_point, σ=0.5)]
	hr_dense = get_hr(kernel, Hamster.Dense(), 1)
	hr_sparse = get_hr(kernel, Hamster.Sparse(), 1)
	expected_matrix = [expected[1] expected[2]; 0.0 0.0]
	@test hr_dense[1] ≈ expected_matrix
	@test Matrix(hr_sparse[1]) ≈ expected_matrix

	set_params!(kernel, [4.0, 5.0])
	@test get_params(kernel) == [4.0, 5.0]
	copy_params!(kernel, Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf))
	@test get_params(kernel) == [2.0, 3.0]
end
