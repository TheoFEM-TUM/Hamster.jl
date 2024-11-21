@testset "Fit model to GaAs" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "Niter", "Optimizer", 300)

    # Test that errer of effective Hamiltonian model for GaAs is sufficiently small
    @test Hamster.get_validate(conf)
    ham_train = EffectiveHamiltonian(conf)
    optim = GDOptimizer(8, 56, conf)
    dl = DataLoader([1], [1], 8, 8, conf)
    ham_val = EffectiveHamiltonian(conf)
    prof = HamsterProfiler(3, conf, printeachiter=100)
    optimize_model!(ham_train, ham_val, optim, dl, prof, conf)
    @test mean(prof.L_train[:, end]) < 0.15
    @test prof.L_val[end] < 0.5 # includes all bands
end