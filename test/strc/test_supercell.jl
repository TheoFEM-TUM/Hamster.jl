path = joinpath(@__DIR__, "test_files")

@testset "Test Config indices" begin
    # Test 1: no validation, no Nconf_min
    conf = get_empty_config()
    set_value!(conf, "Nconf", "Supercell", 100)
    set_value!(conf, "Nconf_min", "Supercell", 1)
    set_value!(conf, "Nconf_max", "Supercell", 1000)
    config_indices, _ = Hamster.get_config_index_sample(conf)
    @test length(unique(config_indices)) == 100
    @test all(i->1 ≤ i ≤ 1000, config_indices)

    # Test 2: no validation, Nconf_min set to 100, no Config
    config_indices, _ = Hamster.get_config_index_sample(Nconf=100, Nconf_min=100, Nconf_max=1000)
    @test length(unique(config_indices)) == 100
    @test all(i->100 ≤ i ≤ 1000, config_indices)

    # Test 3: validation with ratio 0.2, Nconf_min set to 100
    train_indices, val_indices = Hamster.get_config_index_sample(Nconf=100, Nconf_min=75, Nconf_max=1000, val_ratio=0.2, validate=true, val_mode="md", train_mode="md")
    @test length(unique(train_indices)) == 100
    @test length(unique(val_indices)) == 20
    @test all(i->75 ≤ i ≤ 1000, train_indices)
    @test all(i->75 ≤ i ≤ 1000, val_indices)
    @test isempty(intersect(train_indices, val_indices))

    # Test 4: Test no args, train_inds should be one and val_inds empty
    train_indices, val_indices = Hamster.get_config_index_sample()
    @test train_indices == [1]
    @test isempty(val_indices)
    
    # Test 5: If validate and Nconf=1 should return [1] for val_indices
    conf = get_empty_config()
    set_value!(conf, "validate", "Optimizer", true)
    train_indices, val_indices = Hamster.get_config_index_sample(conf)
    @test train_indices == val_indices == [1]

    # Test 6: Test md validation with pc training
    train_indices, val_indices = Hamster.get_config_index_sample(Nconf=10, Nconf_max=100, validate=true, val_mode="md")
    @test train_indices == [1]
    @test length(val_indices) == 10

    # Test 7: Test providing indices
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", "1 3 4")
    train_indices, val_indices = Hamster.get_config_index_sample(conf, train_mode="md")
    @test train_indices == [1, 3, 4]
    @test val_indices == Int64[]

    # Test 8: Test providing val indices and train indices partially
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", "1")
    set_value!(conf, "val_config_inds", "Supercell", "2")
    train_indices, val_indices = Hamster.get_config_index_sample(conf, train_mode="md", val_mode="md", validate=true, Nconf=10, Nconf_max=100, val_ratio=0.2)
    @test train_indices[1] == 1
    @test length(train_indices) == 10
    @test val_indices[1] == 2
    @test length(val_indices) == 2

    # Test 9: Test reading indices from *.dat file
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", joinpath(path, "config_inds.dat"))
    set_value!(conf, "val_config_inds", "Supercell", joinpath(path, "config_inds.dat"))
    train_indices, val_indices = Hamster.get_config_index_sample(conf, train_mode="md", val_mode="md", validate=true)
    @test train_indices == [2, 5, 6]
    @test val_indices == [2, 5, 6]

    # Test 10: Test reading indices from *.h5 file
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", joinpath(path, "config_inds.h5"))
    set_value!(conf, "val_config_inds", "Supercell", joinpath(path, "config_inds.h5"))
    train_indices, val_indices = Hamster.get_config_index_sample(conf, train_mode="md", val_mode="md", validate=true)
    @test train_indices == [2, 5, 6]
    @test val_indices == [2, 5, 6]

    # Test 11: Test splitting indices into chunks (basic functionality)
    indices = collect(1:9)
    @test Hamster.split_indices_into_chunks(indices, 3, rank=0) == [1, 2, 3]
    @test Hamster.split_indices_into_chunks(indices, 3, rank=1) == [4, 5, 6]
    @test Hamster.split_indices_into_chunks(indices, 3, rank=2) == [7, 8, 9]

    # Test 12: Empty list
    indices = []
    @test Hamster.split_indices_into_chunks(indices, 1, rank=0) == []

    # Test 13: Out of range
    indices = [1, 2]
    @test Hamster.split_indices_into_chunks(indices, 3, rank=0) == [1]
    @test Hamster.split_indices_into_chunks(indices, 3, rank=1) == [2]
    @test Hamster.split_indices_into_chunks(indices, 3, rank=2) == []
end

@testset "Multiple Structures from XDATCAR" begin
    # Test 1: Test basic functionality
    conf = get_empty_config()
    set_value!(conf, "XDATCAR", "Supercell", joinpath(path, "XDATCAR_gaas"))
    set_value!(conf, "POSCAR", "Supercell", joinpath(path, "SC_POSCAR_gaas"))
    set_value!(conf, "Nconf", "Supercell", 10)
    set_value!(conf, "Nconf_min", "Supercell", 100)
    set_value!(conf, "Nconf_max", "Supercell", 200)

    poscar = Hamster.read_poscar(Hamster.get_sc_poscar(conf))
    lattice, configs = Hamster.read_xdatcar(Hamster.get_xdatcar(conf), frac=false)

    config_indices, _ = Hamster.get_config_index_sample(conf)
    strcs = get_structures(conf, mode="md", config_indices=config_indices)

    @test length(strcs) == 10

    correct_atom_types = map(eachindex(strcs)) do n
        Hamster.get_ion_types(strcs[n].ions) == poscar.atom_types
    end
    @test all(correct_atom_types)

    rs_ion = Hamster.frac_to_cart(poscar.rs_atom, poscar.lattice)
    Ts = Hamster.frac_to_cart(Hamster.get_translation_vectors(1), lattice)
    correct_atom_positions = Bool[]
    for n in eachindex(strcs), iion in eachindex(strcs[n].ions)
        push!(correct_atom_positions, strcs[n].ions[iion].pos ≈ rs_ion[:, iion])
        
        r_ion = strcs[n].ions[iion].pos - strcs[n].ions[iion].dist
        rmin = findmin([Hamster.normdiff(r_ion, configs[:, iion, config_indices[n]], Ts[:, R]) for R in axes(Ts, 2)])[1]
        push!(correct_atom_positions, rmin ≈ 0.)
    end
    @test all(correct_atom_positions)

    # Test 2: Test empty index list
    strcs = get_structures(conf, mode="md", config_indices=[])
    @test isempty(strcs)
    strcs = get_structures(conf, mode="pc", config_indices=[])
    @test isempty(strcs)
end

@testset "Multiple Structures from h5 file" begin
    # Test 1: Test basic functionality
    conf = get_empty_config()
    set_value!(conf, "XDATCAR", "Supercell", joinpath(path, "xdatcar_gaas.h5"))
    set_value!(conf, "POSCAR", "Supercell", joinpath(path, "SC_POSCAR_gaas"))
    set_value!(conf, "Nconf", "Supercell", 10)
    set_value!(conf, "Nconf_min", "Supercell", 100)
    set_value!(conf, "Nconf_max", "Supercell", 200)

    poscar = Hamster.read_poscar(Hamster.get_sc_poscar(conf))
    lattice, configs = Hamster.read_xdatcar(joinpath(path, "XDATCAR_gaas"), frac=false)

    config_indices, _ = Hamster.get_config_index_sample(conf)
    strcs = get_structures(conf, mode="md", config_indices=config_indices)

    @test length(strcs) == 10

    correct_atom_types = map(eachindex(strcs)) do n
        Hamster.get_ion_types(strcs[n].ions) == poscar.atom_types
    end
    @test all(correct_atom_types)

    rs_ion = Hamster.frac_to_cart(poscar.rs_atom, poscar.lattice)
    Ts = Hamster.frac_to_cart(Hamster.get_translation_vectors(1), lattice)
    correct_atom_positions = Bool[]
    for n in eachindex(strcs), iion in eachindex(strcs[n].ions)
        push!(correct_atom_positions, strcs[n].ions[iion].pos ≈ rs_ion[:, iion])
        
        r_ion = strcs[n].ions[iion].pos - strcs[n].ions[iion].dist
        rmin = findmin([Hamster.normdiff(r_ion, configs[:, iion, config_indices[n]], Ts[:, R]) for R in axes(Ts, 2)])[1]
        push!(correct_atom_positions, rmin ≈ 0.)
    end
    @test all(correct_atom_positions)

    # Test 2: Test empty index list
    strcs = get_structures(conf, mode="md", config_indices=[])
    @test isempty(strcs)
    strcs = get_structures(conf, mode="pc", config_indices=[])
    @test isempty(strcs)
end