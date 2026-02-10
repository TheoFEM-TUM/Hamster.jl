@testset "Option defaults" begin
    # Test 1: test verbosity
    conf = get_empty_config()
    @test Hamster.get_verbosity(conf) == 1
    set_value!(conf, "verbosity", 0)
    @test Hamster.get_verbosity(conf) == 0
    set_value!(conf, "verbosity", "none")
    @test_throws MethodError Hamster.get_verbosity(conf)

    # Test 2: test system
    conf = get_empty_config()
    @test Hamster.get_system(conf) == "unknown"
    set_value!(conf, "system", "GaAs")
    @test Hamster.get_system(conf) == "GaAs" 

    # Test 3: test init_params
    conf = get_empty_config()
    @test Hamster.get_init_params(conf) == "ones"
    set_value!(conf, "init_params", "random")
    @test Hamster.get_init_params(conf) == "random"

    # Test 4: test nthreads_kpoints
    conf = get_empty_config()
    @test Hamster.get_nthreads_kpoints(conf) == Threads.nthreads()
    set_value!(conf, "nthreads_kpoints", 4)
    @test Hamster.get_nthreads_kpoints(conf) == 4
    set_value!(conf, "nthreads_kpoints", 5.3)
    @test_throws InexactError Hamster.get_nthreads_kpoints(conf)

    # Test 5: test nthreads_bands
    conf = get_empty_config()
    @test Hamster.get_nthreads_bands(conf) == Threads.nthreads()
    set_value!(conf, "nthreads_bands", 4)
    @test Hamster.get_nthreads_bands(conf) == 4
    set_value!(conf, "nthreads_bands", 5.3)
    @test_throws InexactError Hamster.get_nthreads_bands(conf)

    # Test 6: test nhamster
    conf = get_empty_config()
    @test Hamster.get_nhamster(conf) == 1
    set_value!(conf, "nhamster", 2)
    @test Hamster.get_nhamster(conf) == 2
    set_value!(conf, "nhamster", 3.3)
    @test_throws InexactError Hamster.get_nhamster(conf)

    # Test 7: test neig
    conf = get_empty_config()
    Hamster.get_neig(conf) == 6
    set_value!(conf, "neig", 8)
    Hamster.get_neig(conf) == 8

    # Test 8: test sp_diag
    conf = get_empty_config()
    @test Hamster.get_sp_mode(conf) isa Hamster.Sparse
    @test Hamster.get_sp_diag(conf) isa Hamster.Dense
    set_value!(conf, "sp_diag", true)
    @test Hamster.get_sp_diag(conf) isa Hamster.Sparse
    
    conf = get_empty_config()
    set_value!(conf, "skip_diag", true)
    @test Hamster.get_sp_diag(conf) isa Hamster.Sparse

    # Test 9: test write_current
    conf = get_empty_config()
    @test Hamster.get_write_hr(conf) == false
    set_value!(conf, "write_current", true)
    @test Hamster.get_write_hr(conf) == true
end

@testset "Optim defaults" begin
    # Test 1: test eig_fit
    conf = get_empty_config()
    @test Hamster.get_eig_fit(conf) == true
    set_value!(conf, "hr_fit", "Optimizer", true)
    @test Hamster.get_eig_fit(conf) == false
    set_value!(conf, "eig_fit", "Optimizer", true)
    @test Hamster.get_eig_fit(conf) == true

    # Test 2: test update TB
    conf = get_empty_config()
    @test Hamster.get_update_tb(conf, 3) == [false, false, false]
    set_value!(conf, "lr", "Optimizer", 0.1)
    @test Hamster.get_update_tb(conf, 3) == [true, true, true]
    set_value!(conf, "update_tb", "Optimizer", "1 3")
    @test Hamster.get_update_tb(conf, 3) == [true, false, true]
end

@testset "SOC defaults" begin
    get_soc = Hamster.get_soc
    
    # Test 1: test soc tag
    conf = get_empty_config()
    @test get_soc(conf) == false
    
    set_value!(conf, "soc", true)
    @test get_soc(conf) == true

    # Test 2: test soc block
    conf = get_empty_config()
    @test get_soc(conf) == false
    set_value!(conf, "update", "SOC", true)
    @test get_soc(conf) == true

    # Test 3: test init_params
    conf = get_empty_config()
    set_value!(conf, "soc", true)
    @test Hamster.get_soc_init_params(conf) == "zeros"
    set_value!(conf, "init_params", "params.dat")
    @test Hamster.get_soc_init_params(conf) == "params.dat"
    set_value!(conf, "init_params", "SOC", "params_soc.dat")
    @test Hamster.get_soc_init_params(conf) == "params_soc.dat"

end

@testset "HyperOpt defaults" begin
    conf = get_empty_config()

    # Test 1: test params default
    @test Hamster.get_hyperopt_params(conf) == String[]

    # Test 2: test single parameter
    set_value!(conf, "params", "HyperOpt", "param1")
    @test Hamster.get_hyperopt_params(conf) == ["param1"]

    # Test 3: test multiple values
    set_value!(conf, "params", "HyperOpt", "param1 param2")
    @test Hamster.get_hyperopt_params(conf) == ["param1", "param2"]

    # Test 4: test lowerbounds default
    @test Hamster.get_hyperopt_lowerbounds(conf) == [0]

    # Test 5: test single value
    set_value!(conf, "lowerbounds", "HyperOpt", 1)
    @test Hamster.get_hyperopt_lowerbounds(conf) == [1]

    # Test 6: test multiple values
    set_value!(conf, "lowerbounds", "HyperOpt", "1 2")
    @test Hamster.get_hyperopt_lowerbounds(conf) == Float64[1, 2]

    # Test 7: test upperbounds default
    @test Hamster.get_hyperopt_upperbounds(conf) == [0]

    # Test 8: test single value
    set_value!(conf, "upperbounds", "HyperOpt", -1)
    @test Hamster.get_hyperopt_upperbounds(conf) == [-1]

    # Test 9: test multiple values
    set_value!(conf, "upperbounds", "HyperOpt", "3 4")
    @test Hamster.get_hyperopt_upperbounds(conf) == Float64[3, 4]

    # Test 10: test stepsizes default
    set_value!(conf, "params", "HyperOpt", "param1 param2")
    @test Hamster.get_hyperopt_stepsizes(conf) == [1e-5, 1e-5]

    # Test 11: test single value
    set_value!(conf, "stepsizes", "HyperOpt", 0.1)
    @test Hamster.get_hyperopt_stepsizes(conf) == [0.1]

    # Test 12: test multiple values
    set_value!(conf, "stepsizes", "HyperOpt", "0.1 0.2")
    @test Hamster.get_hyperopt_stepsizes(conf) == [0.1, 0.2] 
end
