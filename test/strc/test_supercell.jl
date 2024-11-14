path = string(@__DIR__) * "/test_files/"

@testset "Multiple Structures from XDATCAR" begin
    conf = get_empty_config()
    set_value!(conf, "XDATCAR", "Supercell", path*"XDATCAR_gaas")
    set_value!(conf, "POSCAR", "Supercell", path*"SC_POSCAR_gaas")
    set_value!(conf, "Nconf", "Supercell", 10)
    set_value!(conf, "Nconfmin", "Supercell", 100)
    
    poscar = Hamster.read_poscar(Hamster.get_sc_poscar(conf))
    lattice, configs = Hamster.read_xdatcar(Hamster.get_xdatcar(conf), frac=false)

    strcs, config_inds = get_structures(conf)

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
        rmin = findmin([Hamster.normdiff(r_ion, configs[:, iion, config_inds[n]], Ts[:, R]) for R in axes(Ts, 2)])[1]
        push!(correct_atom_positions, rmin ≈ 0.)
    end
    @test all(correct_atom_positions)
end

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