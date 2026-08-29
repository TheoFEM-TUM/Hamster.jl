@testset "Kernel point sorting" begin
	data_points = [
		SVector{8, Float64}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
		SVector{8, Float64}(0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
		SVector{8, Float64}(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
	]
	weights = Int64[10, 20, 30]
	params = [1.0, 2.0, 3.0]
	conf = get_empty_config()
	set_value!(conf, "key_dims", "ML", "1 2")

	@test_throws AssertionError Hamster.sort_by_key(data_points, Int64[10, 20], params, get_empty_config(); key_dims=Int64[1, 2])

	sorted, sorted_weights, sorted_params, ranges = Hamster.sort_by_key(data_points, weights, params, get_empty_config(); key_dims=Int64[1, 2])

	@test sorted == data_points[[2, 1, 3]]
	@test sorted_weights == weights[[2, 1, 3]]
	@test sorted_params == params[[2, 1, 3]]
	@test ranges == Dict((0, 2) => 1:1, (1, 0) => 2:2, (1, 1) => 3:3)

	kernel_points, reordered_params = Hamster.get_sorted_Kernelpoints(data_points, weights, params, conf)
	@test kernel_points.datapoints == data_points[[2, 1, 3]]
	@test kernel_points.weights == weights[[2, 1, 3]]
	@test kernel_points.keys == [(0, 2), (1, 0), (1, 1)]
	@test kernel_points.key_sizes == Dict((0, 2) => 1, (1, 0) => 1, (1, 1) => 1)
	@test reordered_params == params[[2, 1, 3]]
	@test Hamster.verify_kernelpoints(kernel_points, conf)

	bad_datapoints = [
		SVector{8, Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
		SVector{8, Float64}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
	]
	bad_kernel_points = Hamster.Kernelpoints(
		bad_datapoints,
		[1, 1],
		[(0, 0), (1, 0)],
		Dict((0, 0) => 1:1, (1, 0) => 1:1),
		Dict((0, 0) => 1, (1, 0) => 1),
	)
	@test !Hamster.verify_kernelpoints(bad_kernel_points, get_empty_config(); key_dims=Int64[1, 2])

	bad_key_ranges = Dict((0,) => 1:1, (1,) => 2:2)
	@test !Hamster.check_consistency(data_points, [1.0, 2.0], key_ranges=bad_key_ranges, verbose=false)
	@test !Hamster.check_consistency(data_points, [1.0], key_ranges=Dict((0,) => 2:2), verbose=false)
end

@testset "Similarity matrix features" begin
	point = SVector{8, Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	far_point = SVector{8, Float64}(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	data_points = [point, far_point]
	params = [2.0, 3.0]
	kp, _ = Hamster.get_sorted_Kernelpoints(data_points, ones(Int64, 2), params)
    conf = get_empty_config()
    set_value!(conf, "verbosity", 0)
	descriptor = sparse([1, 1], [1, 2], [point, far_point], 1, 2)
	sim_mat = Hamster.get_kernel_features([[descriptor]], kp, 0.1, 0.01, conf = conf)

	@test sim_mat isa Hamster.SimMat
	@test sim_mat.feature_shape == ([(1, 1)], 2)
	@test sim_mat.sim_params == 0.1
	@test length(sim_mat.feature_vec) == 1
	@test sim_mat.feature_vec[1][1][2] == 2
	features = sim_mat.feature_vec[1][1][1]
	@test features[1][1][1] ≈ 1.0
	@test features[1][2] == (1, 1, (0,))
	@test features[2][1][2] ≈ 1.0
	@test features[2][2] == (1, 2, (0,))

	@test Hamster.exp_sim2(point, point, σ=0.1) ≈ 1.0
	@test Hamster.exp_sim_all([point, far_point], point, σ=0.1)[1] ≈ 1.0
	@test Hamster.exp_sim_all([point, far_point], point, σ=0.1)[2] < 0.01

	bad_point = SVector{8, Float64}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	bad_descriptor = sparse([1], [1], [bad_point], 1, 1)
	bad_kp = Hamster.get_sorted_Kernelpoints([SVector{8, Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)], Int64[1], [1.0])
	@test_throws TaskFailedException Hamster.get_kernel_features([[bad_descriptor]], bad_kp[1], 0.1, 0.01; key_dims=Int64[1])
end

@testset "Similarity matrix consistency" begin
	points = [SVector{8, Float64}(zeros(8))]
	@test Hamster.check_consistency(points, [1.0], verbose=false)
	@test !Hamster.check_consistency(points, [1.0, 2.0], verbose=false)
	@test Hamster.check_consistency(points, [1.0], key_ranges=Dict((0,) => 1:1), verbose=false)
	@test !Hamster.check_consistency(points, [1.0], key_ranges=Dict((0,) => 2:2), verbose=false)
end
