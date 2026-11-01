@testset "Fit model to GaAs eigenvalues" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "Niter", "Optimizer", 300)

    # Test that error of effective Hamiltonian model for GaAs is sufficiently small
    @test Hamster.get_validate(conf)
    strc = Structure(conf); basis = Basis(strc, conf)
    ham_train = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    optim = GDOptimizer(8, 56, conf)
    dl = DataLoader(Dict(""=>[1]), Dict(""=>[1]), Dict(""=>8), Dict(""=>8), conf)
    ham_val = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    prof = HamsterProfiler(3, conf, printeachiter=100)
    optimize_model!(ham_train, ham_val, optim, dl, prof, comm, conf)
    @test mean(prof.L_train[:, end]) < 0.15
    @test prof.L_val[end] < 0.5 # includes all bands
end

@testset "Fit model to GaAs eigenvalues (sparse mode)" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_weights", "Optimizer", true)
    set_value!(conf, "sp_mode", true)
    set_value!(conf, "sp_tol", 1e-5)
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "Niter", "Optimizer", 300)

    # Test that error of effective Hamiltonian model for GaAs is sufficiently small
    @test Hamster.get_validate(conf)
    strc = Structure(conf); basis = Basis(strc, conf)
    ham_train = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    optim = GDOptimizer(8, 56, conf)
    dl = DataLoader(Dict(""=>[1]), Dict(""=>[1]), Dict(""=>8), Dict(""=>8), conf)
    ham_val = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    prof = HamsterProfiler(3, conf, printeachiter=100)
    optimize_model!(ham_train, ham_val, optim, dl, prof, comm, conf)
    @test mean(prof.L_train[:, end]) < 0.15
    @test prof.L_val[end] < 0.15 # same loss as train
end

@testset "Fit model to GaAs Hr" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "wannier90_hr.dat"))
    set_value!(conf, "hr_fit", "Optimizer", true)
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "eig_val", "Optimizer", true)
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "Niter", "Optimizer", 300)
    set_value!(conf, "NNaxes", "As", false)
    Hr, Rs, deg = Hamster.read_hrdat(joinpath(path, "wannier90_hr.dat"))

    # Test that error of effective Hamiltonian model for GaAs is sufficiently small
    @test Hamster.get_validate(conf)
    strcs = get_structures(conf, Rs=Rs)
    @test strcs[1].Rs == Rs
    bases = Basis[Basis(strc, conf) for strc in strcs]
    ham_train = EffectiveHamiltonian(strcs, bases, comm, conf, rank=rank, nranks=nranks)
    
    optim = GDOptimizer(conf)
    dl = DataLoader(Dict(""=>[1]), Dict(""=>[1]), Dict(""=>8), Dict(""=>8), conf)
    
    strcs_val = get_structures(conf)
    bases_val = Basis[Basis(strc, conf) for strc in strcs_val]
    ham_val = EffectiveHamiltonian(strcs_val, bases_val, comm, conf, rank=rank, nranks=nranks)

    prof = HamsterProfiler(3, conf, printeachiter=20)
    optimize_model!(ham_train, ham_val, optim, dl, prof, comm, conf)
    @test mean(prof.L_train[:, end]) < mean(prof.L_train[:, 1:5])
    @test prof.L_val[end] < mean(prof.L_val[1:5])
end
@testset "Regularization enters the gradient once over the MPI ranks" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "verbosity", 0)
    strc = Structure(conf); basis = Basis(strc, conf)
    ham = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    optim = GDOptimizer(8, 56, conf)
    data = DataLoader(Dict(""=>[1]), Dict(""=>[1]), Dict(""=>8), Dict(""=>8), conf).train_data[1]
    _, cache = Hamster.forward(ham, 1, optim.loss, data)
    dL_dHr = Hamster.backward(ham, 1, optim.loss, data, cache)
    tb = ham.models[1]
    reg = Hamster.Regularization(0.0, 0.1, 2)
    @test any(!iszero, Hamster.backward(reg, tb.params))

    # Two structures on one rank vs. one structure on each of two ranks (whose gradients are summed)
    serial = Hamster.get_model_gradient(tb, [1, 1], Hamster.rank_regularization(reg, 0), [dL_dHr, dL_dHr])
    distributed = sum(Hamster.get_model_gradient(tb, [1], Hamster.rank_regularization(reg, r), [dL_dHr]) for r in 0:1)
    @test distributed ≈ serial
end
