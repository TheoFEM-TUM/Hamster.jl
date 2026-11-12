@testset "HamiltonianKernelPrecalced smoke" begin
    point = SVector{9, Float64}(zeros(9))
    second_point = SVector{9, Float64}(1.0, zeros(8)...)
    descriptor = sparse([1, 1], [1, 2], [point, second_point], 2, 2)
    conf = get_empty_config()
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "key_dims", "ML", 0)  # no key dimensions: all data points share the key (0,)
    base_kernel = HamiltonianKernel([2.0, 3.0], [point, second_point], 0.5, [[descriptor]], true)
    kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf)

    @test get_params(kernel) == [2.0, 3.0]
    @test kernel.kp.datapoints == [point, second_point]
    @test kernel.kp.key_ranges == Dict((0,) => 1:2)
    @test kernel.kp.weights == [1, 1]
    @test kernel.sms[1].sim_params == 0.5

    expected = [2.0 * Hamster.exp_sim(point, point, σ=0.5) + 3.0 * Hamster.exp_sim(second_point, point, σ=0.5),
                2.0 * Hamster.exp_sim(point, second_point, σ=0.5) + 3.0 * Hamster.exp_sim(second_point, second_point, σ=0.5)]
    expected_matrix = [expected[1] expected[2]; 0.0 0.0]
    @test get_hr(kernel, Hamster.Dense(), 1)[1] ≈ expected_matrix
    @test Matrix(get_hr(kernel, Hamster.Sparse(), 1)[1]) ≈ expected_matrix

    set_params!(kernel, [4.0, 5.0])
    @test get_params(kernel) == [4.0, 5.0]
    copy_params!(kernel, Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf))
    @test get_params(kernel) == [2.0, 3.0]

    # The multiplicities of the data points weight the gradient
    @test Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf, weights=[3, 1]).kp.weights == [3, 1]
end

@testset "Precalced ML Parameter" begin
    conf = get_empty_config()
    set_value!(conf, "rcut", "ML", 5.0)
    set_value!(conf, "sim_params", "ML", 0.2)
    set_value!(conf, "env_scale", "ML", 0.8)
    set_value!(conf, "apply_distortion", "ML", false)
    set_value!(conf, "verbosity", 0)

    # The first three dimensions are the key of a data point (key_dims = 1 2 3)
    data_points = [SVector{9, Float64}(1, 31, isodd(i) ? 31 : 33, rand(6)...) for i in 1:10]
    params = rand(10)
    base_kernel = HamiltonianKernel(params, data_points, 0.2, [], false)
    kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, String[], conf)
    @test sort(collect(zip(kernel.params, kernel.kp.datapoints))) == sort(collect(zip(params, data_points)))

    # The parameters are written in the order of the sorted kernel points
    write_params(kernel, conf)
    params_1, data_points_1 = Hamster.read_ml_params(conf)
    @test params_1 == kernel.params
    @test data_points_1 == kernel.kp.datapoints
    rm("ml_params.dat")
end

@testset "CsPbBr3 precalced and on-the-fly kernels" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_CsPbBr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "init_params", "ML", "rand")
    set_value!(conf, "init_params", "zeros")
    set_value!(conf, "verbosity", 0)
    strc = Structure(conf)
    basis = Basis(strc, conf)
    model = TBModel([strc], [basis], comm, conf)

    # sim_mat = true (default): precalculated similarity matrix, sim_mat = false: similarities to all data points on the fly
    @test HamiltonianKernel([strc], [basis], model, comm, conf) isa Hamster.HamiltonianKernelPrecalced
    set_value!(conf, "sim_mat", "ML", false)
    @test !Hamster.get_ml_sim_mat(conf)
    base_kernel = HamiltonianKernel([strc], [basis], model, comm, conf)
    @test base_kernel isa HamiltonianKernel

    kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf)
    for k in (base_kernel, kernel)
        Hk = get_hamiltonian(get_hr(k, Hamster.Dense(), 1), strc.Rs, zeros(3, 1))
        @test abs(sum(Hk[1] .- Hermitian(Hk[1]))) < 1e-7
    end

    # Without key buckets and similarity cutoff, the precalculated similarities reproduce the on-the-fly kernel
    set_value!(conf, "key_dims", "ML", 0)
    set_value!(conf, "sim_tol", "ML", 0.0)
    kernel = Hamster.HamiltonianKernelPrecalced(base_kernel, 0, ["test"], conf)
    for mode in (Hamster.Dense(), Hamster.Sparse())
        @test all(Matrix.(get_hr(base_kernel, mode, 1)) .≈ Matrix.(get_hr(kernel, mode, 1)))
    end
    base_kernel.update = kernel.update = true
    dL_dHr = [[sprand(ComplexF64, size(H)..., 0.5) .+ H for H in get_hr(base_kernel, Hamster.Sparse(), 1)]]
    reg = Hamster.Regularization(conf)
    perm = [findfirst(==(x), base_kernel.data_points) for x in kernel.kp.datapoints]
    @test Hamster.get_model_gradient(kernel, [1], reg, dL_dHr) ≈ Hamster.get_model_gradient(base_kernel, [1], reg, dL_dHr)[perm]
end

@testset "On-the-fly kernel in the device code path" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_CsPbBr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "init_params", "ML", "rand")
    set_value!(conf, "init_params", "zeros")
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "sim_mat", "ML", false)
    strc = Structure(conf)
    basis = Basis(strc, conf)
    ham = EffectiveHamiltonian([strc], [basis], comm, conf, device=Hamster.ArrayDevice(Array))
    kernel = ham.models[2]
    @test kernel isa HamiltonianKernel
    lmap = ham.device_ctx.maps[1]
    block = only(b for b in lmap.blocks if b.model_index == 2)
    @test block isa Hamster.DeviceKernelBlock

    # Eigenvalues: device path (Gaussian kernel on the device) vs. host path (get_hr)
    ks = [0.0 0.5 0.25; 0.0 0.0 0.25; 0.0 0.5 0.0]
    Es_device = sort(first(Hamster.device_diagonalize(ham, 1, ks)), dims=1)
    Hr = reduce((x, y) -> x .+ y, [get_hr(model, Hamster.Dense(), 1) for model in ham.models])
    Es_host = reduce(hcat, [eigvals(Hermitian(Matrix(H))) for H in get_hamiltonian(Hr, ham.Rs[1], ks)])
    @test Es_device ≈ Es_host atol=1e-8

    # Parameter gradient of the device kernel block vs. host gradient for the same dL/dHr
    kernel.update = true
    dL_dHr = [[sprand(ComplexF64, size(H)..., 0.5) .+ H for H in get_hr(kernel, Hamster.Sparse(), 1)]]
    G = [dL_dHr[1][lmap.Ridx[e]][lmap.I[e], lmap.J[e]] for e in eachindex(lmap.I)]
    reg = Hamster.Regularization(conf, lambda=0.0)
    @test Hamster.parameter_gradient(block, G) ≈ Hamster.get_model_gradient(kernel, [1], reg, dL_dHr)
end
