@testset "DataLoader PC Eigs" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_empty_config()
    kp, Es = Hamster.read_eigenval(joinpath(path, "EIGENVAL_gaas"), 8)
    
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))

    # Test 1: test PC training data with no validation
    dl = DataLoader([1], [1], 8, 8, conf)
    @test typeof(dl.val_data) == typeof(dl.train_data)
    @test length(dl.val_data) == 0
    @test length(dl.train_data) == 1
    
    @test dl.train_data[1].kp ≈ kp
    @test dl.train_data[1].Es ≈ Es

    # Test 2: test PC training data with validation
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    dl = DataLoader([1], [1], 8, 8, conf)
    @test length(dl.val_data) == 1
    @test dl.val_data[1].kp ≈ kp
    @test dl.val_data[1].Es ≈ Es

    # Test 3: test get nk and neig
    @test Hamster.get_neig_and_nk(dl.train_data) == (8, 56)
end

@testset "DataLoader MD Eigs" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_empty_config()
    kp = h5read(joinpath(path, "eigenvalues_md.h5"), "kpoints")
    Es = h5read(joinpath(path, "eigenvalues_md.h5"), "eigenvalues")

    set_value!(conf, "train_data", "Optimizer", joinpath(path, "eigenvalues_md.h5"))
    set_value!(conf, "train_mode", "Optimizer", "MD")

    # Test 1: test MD training data with no validation
    dl = DataLoader([2, 3, 4], [1], 8, 32, conf)
    
    @test typeof(dl.val_data) == typeof(dl.train_data)
    @test length(dl.val_data) == 0
    @test length(dl.train_data) == 3
    @test all([data.kp ≈ kp for data in dl.train_data])
    @test all([dl.train_data[i1].Es ≈ Es[:, :, i2] for (i1, i2) in zip(eachindex(dl.train_data), [2, 3, 4])])

    # Test 2: test MD training data with pc validation
    kp2, Es2 = Hamster.read_eigenval(joinpath(path, "EIGENVAL_gaas"), 8)
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))
    dl = DataLoader([2, 3, 4], [1], 8, 32, conf)
    
    @test typeof(dl.val_data) == typeof(dl.train_data)
    @test length(dl.val_data) == 1
    @test length(dl.train_data) == 3
    @test dl.val_data[1].kp ≈ kp2
    @test dl.val_data[1].Es ≈ Es2

    # Test 3: test MD training data MD validation
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "eigenvalues_md.h5"))
    set_value!(conf, "val_mode", "Optimizer", "MD")
    dl = DataLoader([2, 3, 4], [1, 6, 8], 8, 32, conf)

    @test typeof(dl.val_data) == typeof(dl.train_data)
    @test length(dl.val_data) == 3
    @test length(dl.train_data) == 3
    @test all([data.kp ≈ kp for data in dl.val_data])
    @test all([dl.val_data[i1].Es ≈ Es[:, :, i2] for (i1, i2) in zip(eachindex(dl.val_data), [1, 6, 8])])
end

@testset "DataLoader Mixed Eigs" begin

end

@testset "DataLoader PC Hr" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_empty_config()
    Hr, Rs, deg = Hamster.read_hrdat(joinpath(path, "wannier90_hr.dat"))
    
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "wannier90_hr.dat"))
    set_value!(conf, "hr_fit", "Optimizer", true)

    # Test 1: test PC training data with no validation
    dl = DataLoader([], [], 8, 8, conf)
    @test all([dl.train_data[1].Hr[R] == Hr[R] for R in eachindex(Hr)])
    @test length(dl.val_data) == 0
    @test typeof(dl.train_data) == typeof(dl.val_data)

    # Test 2: test PC training data with validation
    set_value!(conf, "val_data", "Optimizer", joinpath(path, "wannier90_hr.dat"))
    dl = DataLoader([], [], 8, 8, conf)
    @test length(dl.val_data) == 1
    @test all([dl.val_data[1].Hr[R] == Hr[R] for R in eachindex(Hr)])
    
    # Test 3: test get nk and neig
    @test Hamster.get_neig_and_nk(dl.train_data) == (0, 0)
end

@testset "DataLoader MD Hr" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_empty_config()
    Hr = h5read(joinpath(path, "md_hr.h5"), "Hr")
    
    set_value!(conf, "train_data", "Optimizer", joinpath(path, "md_hr.h5"))
    set_value!(conf, "hr_fit", "Optimizer", true)
    set_value!(conf, "train_mode", "Optimizer", "MD")

    dl = DataLoader([1, 3], [], 8, 8, conf)
    @test length(dl.train_data) == 2
    @test all([dl.train_data[1].Hr[R] == Hr[:, :, R, 1] for R in axes(Hr, 3)])
    @test all([dl.train_data[2].Hr[R] == Hr[:, :, R, 3] for R in axes(Hr, 3)])
end