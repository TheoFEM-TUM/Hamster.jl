@testset "Test Config indices" begin
    # Test 1: no validation, no Nconf_min
    conf = get_empty_config()
    set_value!(conf, "Nconf", "Supercell", 100)
    set_value!(conf, "Nconf_min", "Supercell", 1)
    config_indices, _ = Hamster.get_config_index_sample(1000, conf)
    @test length(unique(config_indices)) == 100
    @test all(i->1 ≤ i ≤ 1000, config_indices)

    # Test 2: no validation, Nconf_min set to 100, no Config
    config_indices, _ = Hamster.get_config_index_sample(1000, Nconf=100, Nconf_min=100)
    @test length(unique(config_indices)) == 100
    @test all(i->100 ≤ i ≤ 1000, config_indices)

    # Test 3: validation with ratio 0.2, Nconf_min set to 100
    train_indices, val_indices = Hamster.get_config_index_sample(1000, Nconf=100, Nconf_min=75, val_ratio=0.2)
    @test length(unique(train_indices)) == 100
    @test length(unique(val_indices)) == 20
    @test all(i->75 ≤ i ≤ 1000, train_indices)
    @test all(i->75 ≤ i ≤ 1000, val_indices)
    @test isempty(intersect(train_indices, val_indices))
end