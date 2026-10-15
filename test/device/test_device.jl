using JLArrays

device_path = joinpath(@__DIR__, "..", "calc", "test_files")
device_soc_path = joinpath(@__DIR__, "..", "soc", "test_files")

function get_device_test_config(; sp_mode=false, sp_diag=false, sim_mat=false)
    conf = get_config(filename=joinpath(device_path, "hconf_ml"))
    set_value!(conf, "poscar", joinpath(device_path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(device_path, "rllm.dat"))
    set_value!(conf, "init_params", joinpath(device_path, "params.dat"))
    set_value!(conf, "init_params", "ML", "rand")
    set_value!(conf, "update", "ML", true)
    set_value!(conf, "update_tb", "Optimizer", true)
    set_value!(conf, "sp_mode", sp_mode)
    set_value!(conf, "sp_diag", sp_diag)
    set_value!(conf, "sim_mat", "ML", sim_mat)
    return conf
end

function get_soc_device_test_config()
    conf = get_config(filename=joinpath(device_soc_path, "hconf_gaas"))
    set_value!(conf, "rllm_file", joinpath(device_soc_path, "rllm_gaas.dat"))
    set_value!(conf, "poscar", joinpath(device_soc_path, "POSCAR_gaas"))
    set_value!(conf, "init_params", joinpath(device_soc_path, "params.dat"))
    set_value!(conf, "init_params", "SOC", joinpath(device_soc_path, "params.dat"))
    set_value!(conf, "update", "SOC", true)
    set_value!(conf, "verbosity", 0)
    return conf
end

# Build the CPU reference and device Hamiltonians for the same structure. Seeding before each construction
# ensures that both models sample the same ML data points and random ML parameters.
function get_device_test_hams(conf, device)
    strc = Structure(conf); basis = Basis(strc, conf)
    Hamster.Random.seed!(1234)
    ham_cpu = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    Hamster.Random.seed!(1234)
    ham_dev = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks, device=device)
    return ham_cpu, ham_dev
end

# Dense real-space Hamiltonian from the device entries
function get_dense_device_hr(ham, index)
    ctx = ham.device_ctx; lmap = ctx.maps[index]
    hr = Array(Hamster.get_device_hr(lmap, Hamster.device_params(ham, ctx)))
    I, J, Ridx = Array(lmap.I), Array(lmap.J), Array(lmap.Ridx)
    Hr = [zeros(ComplexF64, lmap.Nε, lmap.Nε) for _ in 1:lmap.NR]
    for e in eachindex(hr)
        Hr[Ridx[e]][I[e], J[e]] += hr[e]
    end
    return Hr
end

maxdiff(As, Bs) = maximum(maximum(abs.(A .- B)) for (A, B) in zip(As, Bs))

function get_device_loss(ham, index, loss, data)
    return Hamster.forward(ham, index, loss, data)[1]
end

# Generic k-points without degenerate eigenvalues, such that eigenvalue gradients are well-defined
ks_device_test = [0.1 0.3 0.45; 0.2 0.1 0.15; 0.05 0.4 0.3]

@testset "Device linear map reproduces Hr and Hk" begin
    for (sp_mode, sim_mat) in [(false, false), (true, false), (true, true)], device in [ArrayDevice(Array), ArrayDevice(JLArray)]
        conf = get_device_test_config(sp_mode=sp_mode, sim_mat=sim_mat)
        ham_cpu, ham_dev = get_device_test_hams(conf, device)
        @test ham_dev.device_ctx isa Hamster.DeviceContext
        @test isnothing(ham_cpu.device_ctx)

        Hr_cpu = Matrix.(get_hr(ham_cpu, 1))
        Hr_dev = get_dense_device_hr(ham_dev, 1)
        @test maxdiff(Hr_cpu, Hr_dev) < 1e-10

        Hk_cpu = Matrix.(get_hamiltonian(ham_cpu, 1, ks_device_test))
        Hk_dev = get_hamiltonian(ham_dev, 1, ks_device_test)
        @test Hk_dev isa Hamster.DeviceHk
        @test Hamster.hamiltonian_dim(Hk_dev) == size(Hk_cpu[1], 1)
        @test maxdiff(Hk_cpu, Hamster.host_hamiltonians(Hk_dev)) < 1e-10
        @test maxdiff(Hk_cpu, [Array(Hamster.densify(Hk_dev))[:, :, k] for k in 1:3]) < 1e-10

        Es_cpu, _ = diagonalize(Hk_cpu)
        Es_dev, vs_dev = diagonalize(Hk_dev)
        @test Es_cpu ≈ Es_dev atol=1e-8
        @test vs_dev isa Array{ComplexF64, 3}
    end
end

@testset "Device gradients match CPU gradients" begin
    for sim_mat in [false, true], device in [ArrayDevice(Array), ArrayDevice(JLArray)]
        conf = get_device_test_config(sp_mode=true, sim_mat=sim_mat)
        ham_cpu, ham_dev = get_device_test_hams(conf, device)
        loss = Loss(2); reg = Regularization(conf)

        Es, _ = diagonalize(get_hamiltonian(ham_cpu, 1, ks_device_test))
        data = Hamster.EigData(ks_device_test, Es .+ 0.1 .* randn(size(Es)))

        L_cpu, cache_cpu = Hamster.forward(ham_cpu, 1, loss, data)
        L_dev, cache_dev = Hamster.forward(ham_dev, 1, loss, data)
        @test L_cpu ≈ L_dev

        dL_dHr = Hamster.backward(ham_cpu, 1, loss, data, cache_cpu, conf)
        g_dev = Hamster.backward(ham_dev, 1, loss, data, cache_dev, conf)
        @test g_dev isa Hamster.DeviceParamGradient
        for (model_index, model) in enumerate(ham_cpu.models)
            grad_cpu = Hamster.get_model_gradient(model, [1], reg, [dL_dHr])
            grad_dev = Hamster.get_model_gradient(ham_dev.models[model_index], [1], reg, Hamster.model_gradient_input([g_dev], model_index))
            @test grad_cpu ≈ grad_dev atol=1e-8
        end
    end
end

@testset "Device gradients with SOC match finite differences" begin
    conf = get_soc_device_test_config()
    ham_cpu, ham_dev = get_device_test_hams(conf, ArrayDevice(JLArray))
    @test maxdiff(Matrix.(get_hr(ham_cpu, 1)), get_dense_device_hr(ham_dev, 1)) < 1e-10

    loss = Loss(2)
    Es, _ = diagonalize(get_hamiltonian(ham_cpu, 1, ks_device_test))
    data = Hamster.EigData(ks_device_test, Es .+ 0.1 .* randn(size(Es)))
    _, cache = Hamster.forward(ham_dev, 1, loss, data)
    g_dev = Hamster.backward(ham_dev, 1, loss, data, cache)

    for (model_index, model) in enumerate(ham_dev.models)
        params = get_params(model)
        for p in unique(round.(Int64, range(1, length(params), length=4)))
            p0 = params[p]; δ = 1e-6
            params[p] = p0 + δ; L_plus = get_device_loss(ham_dev, 1, loss, data)
            params[p] = p0 - δ; L_minus = get_device_loss(ham_dev, 1, loss, data)
            params[p] = p0
            @test g_dev.grads[model_index][p] ≈ (L_plus - L_minus) / 2δ atol=1e-5
        end
    end
end

@testset "Device sparse diagonalization" begin
    conf = get_device_test_config(sp_mode=true, sp_diag=true)
    set_value!(conf, "neig", 4)
    set_value!(conf, "eig_target", 0.5)
    set_value!(conf, "diag_method", "krylov-schur")
    ham_cpu, ham_dev = get_device_test_hams(conf, ArrayDevice(JLArray))

    Es_full, _ = diagonalize(Matrix.(get_hamiltonian(ham_cpu, 1, ks_device_test)))
    Hk_dev = get_hamiltonian(ham_dev, 1, ks_device_test)
    @test Hamster.host_hamiltonians(Hk_dev) isa Vector{<:SparseMatrixCSC}
    Es_dev, vs_dev = diagonalize(Hk_dev, Neig=4, target=0.5, method="krylov-schur")
    for k in axes(Es_full, 2)
        Es_closest = sort(Es_full[sortperm(abs.(Es_full[:, k] .- 0.5))[1:4], k])
        @test Es_dev[:, k] ≈ Es_closest atol=1e-6
    end
    @test issorted(Es_dev[:, 1])
    @test size(vs_dev) == (size(Es_full, 1), 4, 3)
end

@testset "Device training step matches CPU training step" begin
    for device in [ArrayDevice(Array), ArrayDevice(JLArray)]
        conf = get_device_test_config(sp_mode=true)
        ham_cpu, ham_dev = get_device_test_hams(conf, device)
        Es, _ = diagonalize(get_hamiltonian(ham_cpu, 1, ks_device_test))
        train_data = [Hamster.EigData(ks_device_test, Es .+ 0.1 .* randn(size(Es)))]
        prof_cpu = HamsterProfiler(3, conf); prof_dev = HamsterProfiler(3, conf)
        optim_cpu = GDOptimizer(size(Es)..., conf); optim_dev = GDOptimizer(size(Es)..., conf)

        Hamster.train_step!(ham_cpu, [1], optim_cpu, train_data, prof_cpu, 1, 1, comm, conf)
        Hamster.train_step!(ham_dev, [1], optim_dev, train_data, prof_dev, 1, 1, comm, conf)
        @test prof_cpu.L_train[1, 1] ≈ prof_dev.L_train[1, 1]
        for (model_cpu, model_dev) in zip(ham_cpu.models, ham_dev.models)
            @test get_params(model_cpu) ≈ get_params(model_dev) atol=1e-10
        end
    end
end

@testset "Device selection" begin
    conf = get_empty_config()
    @test Hamster.init_device(conf) isa CPUDevice
    @test Hamster.uses_device_path(Hamster.init_device(conf)) == false
    set_value!(conf, "device", "tpu")
    @test_throws ErrorException Hamster.init_device(conf)
    if isempty(methods(Hamster.gpu_device))
        set_value!(conf, "device", "gpu")
        @test_throws ErrorException Hamster.init_device(conf)
    end
end
