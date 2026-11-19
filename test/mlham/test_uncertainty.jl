@testset "GP posterior variance" begin
    d0 = SVector{8}(zeros(8))
    x = SVector{8}(fill(0.3, 8))
    gp = Hamster.GPVariance([d0], 0.1, 1e-3)
    # Single data point: var(x) = 1 - k(x, d0)^2 / (k(d0,d0) + λ) = 1 - k(x, d0)^2 / (1 + λ)
    @test Hamster.gp_variance(gp, x) ≈ 1 - exp(-Hamster.normdiff(x, d0)^2 / (2 * 0.1^2))^2 / 1.001
    @test Hamster.gp_variance(gp, d0) ≈ 1 - 1 / 1.001
    # Far away from all data points the variance approaches k(x,x) = 1
    @test Hamster.gp_variance(gp, SVector{8}(fill(100.0, 8))) ≈ 1.0 atol=1e-12
    # gp_append (incremental Cholesky) matches a fresh factorization
    d1 = SVector{8}(fill(0.5, 8))
    gp2 = Hamster.gp_append(gp, d1)
    gp_fresh = Hamster.GPVariance([d0, d1], 0.1, 1e-3)
    for xx in (x, d0, d1, SVector{8}(fill(0.1, 8)))
        @test Hamster.gp_variance(gp2, xx) ≈ Hamster.gp_variance(gp_fresh, xx)
    end
    @test Hamster.gp_variances(gp_fresh, [x, d0]) ≈ [Hamster.gp_variance(gp_fresh, x), Hamster.gp_variance(gp_fresh, d0)]
    @test all(v -> 0 ≤ v ≤ 1, Hamster.gp_variances(gp_fresh, [x, d0, SVector{8}(fill(2.0, 8))]))
    @test_throws ErrorException Hamster.GPVariance([d0], 0.1, 0.0)
end

@testset "Uncertainty sampling of descriptors" begin
    X = rand(2, 200); X[:, 101:200] .+= 10.0  # two well-separated clusters
    sel = Hamster.sample_structure_descriptors(X, Ncluster=2, Npoints=4, ml_sampling="uncertainty", sim_params=0.5, noise=1e-3)
    @test length(sel) == 4 && length(unique(sel)) == 4
    @test any(s -> s[1] < 5, sel) && any(s -> s[1] > 5, sel)  # points from both clusters
    sel2 = Hamster.sample_structure_descriptors(X, Ncluster=2, Npoints=4, ml_sampling="u", sim_params=0.5, noise=1e-3)
    @test length(sel2) == 4 && length(unique(sel2)) == 4
    @test_throws ErrorException Hamster.sample_structure_descriptors(X, Ncluster=2, Npoints=4, ml_sampling="bogus")
    # Uncertainty sampling picks descriptors with a larger GP variance w.r.t. the cluster centroids than random sampling
    centroids = [mean(X[:, 1:100], dims=2)[:, 1], mean(X[:, 101:200], dims=2)[:, 1]]
    gp = Hamster.GPVariance([SVector{2, Float64}(c) for c in centroids], 0.5, 1e-3)
    Xsel = reduce(hcat, sel)
    Xrand = reduce(hcat, Hamster.sample_structure_descriptors(X, Ncluster=2, Npoints=4, ml_sampling="random"))
    @test mean(Hamster.gp_variances(gp, Xsel)) > mean(Hamster.gp_variances(gp, Xrand))
end

@testset "get_uncertainty of a kernel model" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_CsPbBr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "init_params", "ML", "rand")
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "npoints", "ML", 40)
    set_value!(conf, "ncluster", "ML", 10)
    strc = Structure(conf)
    basis = Basis(strc, conf)
    model = TBModel([strc], [basis], comm, conf)
    kernel = HamiltonianKernel([strc], [basis], model, comm, conf, write_summary=false)

    u, ū = Hamster.get_uncertainty(kernel, 1, noise=1e-3)
    @test length(u) == length(basis)
    @test all(>=(0), u) && ū ≥ 0
    ua = Hamster.atom_uncertainties(u, basis)
    @test length(ua) == length(strc.ions)
    @test sum(ua) ≈ sum(u)

    # A kernel whose data points are the descriptors of the structure themselves: per-element variance ≈ λ/(1+λ) everywhere
    all_desc = [x for R in eachindex(kernel.structure_descriptors[1]) for x in findnz(kernel.structure_descriptors[1][R])[3]]
    gp_dense = Hamster.GPVariance(all_desc, kernel.sim_params, 1e-3)
    @test all(<(0.01), Hamster.gp_variances(gp_dense, all_desc))
    kernel_dense = HamiltonianKernel(zeros(length(all_desc)), all_desc, kernel.sim_params, kernel.structure_descriptors, false)
    _, ū_dense = Hamster.get_uncertainty(kernel_dense, 1, noise=1e-3)
    @test ū_dense < ū

    # The precalced kernel gives the same uncertainty (up to similarities below sim_tol)
    kernel_pre = Hamster.HamiltonianKernelPrecalced(kernel, 0, [""], conf)
    _, ū_pre = Hamster.get_uncertainty(kernel_pre, 1, noise=1e-3)
    @test ū_pre ≈ ū rtol=0.05

    # EffectiveHamiltonian interface (dispatches to the kernel model in ham.models)
    ham = EffectiveHamiltonian([strc], [basis], comm, conf, write_summary=false)
    u3, ū3 = Hamster.get_uncertainty(ham, 1, noise=1e-3)
    u4, ū4 = Hamster.get_uncertainty(ham.models[2], 1, noise=1e-3)
    @test u3 == u4 && ū3 == ū4
end
