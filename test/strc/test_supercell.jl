path = joinpath(@__DIR__, "test_files")

@testset "Test Config indices" begin
    # Test 1: no validation, no Nconf_min
    conf = get_empty_config()
    set_value!(conf, "Nconf", "Supercell", 100)
    set_value!(conf, "Nconf_min", "Supercell", 1)
    set_value!(conf, "Nconf_max", "Supercell", 1000)
    config_indices, _ = Hamster.get_config_index_sample("", conf)
    @test length(unique(config_indices)) == 100
    @test all(i->1 ≤ i ≤ 1000, config_indices)

    # Test 2: no validation, Nconf_min set to 100, no Config
    config_indices, _ = Hamster.get_config_index_sample(""; Nconf=100, Nconf_min=100, Nconf_max=1000)
    @test length(unique(config_indices)) == 100
    @test all(i->100 ≤ i ≤ 1000, config_indices)

    # Test 3: validation with ratio 0.2, Nconf_min set to 100
    train_indices, val_indices = Hamster.get_config_index_sample(""; Nconf=100, Nconf_min=75, Nconf_max=1000, val_ratio=0.2, validate=true, val_mode="md", train_mode="md")
    @test length(unique(train_indices)) == 100
    @test length(unique(val_indices)) == 20
    @test all(i->75 ≤ i ≤ 1000, train_indices)
    @test all(i->75 ≤ i ≤ 1000, val_indices)
    @test isempty(intersect(train_indices, val_indices))

    # Test 4: Test no args, train_inds should be one and val_inds empty
    train_indices, val_indices = Hamster.get_config_index_sample("")
    @test train_indices == [1]
    @test isempty(val_indices)
    
    # Test 5: If validate and Nconf=1 should return [1] for val_indices
    conf = get_empty_config()
    set_value!(conf, "validate", "Optimizer", true)
    train_indices, val_indices = Hamster.get_config_index_sample("", conf)
    @test train_indices == val_indices == [1]

    # Test 6: Test md validation with pc training
    train_indices, val_indices = Hamster.get_config_index_sample(""; Nconf=10, Nconf_max=100, validate=true, val_mode="md")
    @test train_indices == [1]
    @test length(val_indices) == 10

    # Test 7: Test providing indices
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", "1 3 4")
    train_indices, val_indices = Hamster.get_config_index_sample("", conf, train_mode="md")
    @test train_indices == [1, 3, 4]
    @test val_indices == Int64[]

    # Test 8: Test providing val indices and train indices partially
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", "1")
    set_value!(conf, "val_config_inds", "Supercell", "2")
    train_indices, val_indices = Hamster.get_config_index_sample("", conf, train_mode="md", val_mode="md", validate=true, Nconf=10, Nconf_max=100, val_ratio=0.2)
    @test train_indices[1] == 1
    @test length(train_indices) == 10
    @test val_indices[1] == 2
    @test length(val_indices) == 2

    # Test 9: Test reading indices from *.dat file
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", joinpath(path, "config_inds.dat"))
    set_value!(conf, "val_config_inds", "Supercell", joinpath(path, "config_inds.dat"))
    train_indices, val_indices = Hamster.get_config_index_sample("", conf, train_mode="md", val_mode="md", validate=true)
    @test train_indices == [2, 5, 6]
    @test val_indices == [2, 5, 6]

    # Test 10: Test reading indices from *.h5 file
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", joinpath(path, "config_inds.h5"))
    set_value!(conf, "val_config_inds", "Supercell", joinpath(path, "config_inds.h5"))
    train_indices, val_indices = Hamster.get_config_index_sample("", conf, train_mode="md", val_mode="md", validate=true)
    @test train_indices == [2, 5, 6]
    @test val_indices == [2, 5, 6]

    # Test 10b: Partially provided train indices are topped up without duplicates
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", "1 2 3")
    train_indices, _ = Hamster.get_config_index_sample("", conf, train_mode="md", Nconf=10, Nconf_min=1, Nconf_max=10)
    @test train_indices[1:3] == [1, 2, 3]
    @test sort(train_indices) == collect(1:10)

    # Test 10c: More provided val indices than Nconf * val_ratio does not throw
    conf = get_empty_config()
    set_value!(conf, "val_config_inds", "Supercell", "4 5 6")
    train_indices, val_indices = Hamster.get_config_index_sample("", conf, train_mode="md", val_mode="md", validate=true, Nconf=5, Nconf_max=100, val_ratio=0.2)
    @test length(train_indices) == 5
    @test val_indices == [4, 5, 6]

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

    # Test 14: test index splitting with system dict
    indices = Dict(
        "A" => [1, 2, 3, 4],
        "B" => [10, 11],
        "C" => [100, 101, 102]
    )

    chunks = [Hamster.split_indices_into_chunks(indices, 2; rank=i) for i in 0:1]

    # Check that all chunks together reproduce original data
    combined = Dict{String, Vector{Int}}()
    for chunk in chunks
        for (sys, vals) in chunk
            push!(get!(combined, sys, Int[]), vals...)
        end
    end
    for (sys, vals) in indices
        @test sort(combined[sys]) == sort(vals)
    end

    chunks = [Hamster.split_indices_into_chunks(indices, 3; rank=i) for i in 0:2]
    total_configs = sum(length(v) for chunk in chunks for v in values(chunk))
    @test total_configs == sum(length(v) for v in values(indices))

    empty_indices = Dict{String, Vector{Int}}()
    res = Hamster.split_indices_into_chunks(empty_indices, 4; rank=0)
    @test res == Dict{String, Vector{Int}}()

    res = Hamster.split_indices_into_chunks(indices, 2; rank=5)
    @test res == Dict{String, Vector{Int}}()

    # For a simple input, verify rough balance
    simple = Dict("X" => collect(1:10))
    chunks = [Hamster.split_indices_into_chunks(simple, 3; rank=i) for i in 0:2]
    sizes = [sum(length(v) for v in values(c)) for c in chunks]
    @test maximum(sizes) - minimum(sizes) ≤ 1
end

@testset "get_number_of_bands_per_structure" begin
    # Mock input data
    bases = [
        rand(10),  # system A, config 1
        rand(10),  # system A, config 2
        rand(8),   # system B, config 1
        rand(8)    # system B, config 2
    ]

    indices = OrderedDict(
        "A" => [1, 2],
        "B" => [3, 4]
    )

    # --- Test 1: Basic case (no SOC) ---
    result = Hamster.get_number_of_bands_per_structure(bases, indices)
    @test result == Dict("A" => 10, "B" => 8)

    # --- Test 2: With SOC doubling ---
    result_soc = Hamster.get_number_of_bands_per_structure(bases, indices; soc=true)
    @test result_soc == Dict("A" => 20, "B" => 16)

    # --- Test 3: Inconsistent basis lengths should trigger an assertion ---
    bases_bad = [
        rand(10),  # system A, config 1
        rand(12),  # system A, config 2 (inconsistent)
        rand(8),
        rand(8)
    ]
    @test_throws AssertionError Hamster.get_number_of_bands_per_structure(bases_bad, indices)

    # --- Test 4: Single system, single config ---
    bases_single = [rand(5)]
    indices_single = Dict("C" => [1])
    result_single = Hamster.get_number_of_bands_per_structure(bases_single, indices_single)
    @test result_single == Dict("C" => 5)

    # --- Test 5: SOC + single system ---
    result_single_soc = Hamster.get_number_of_bands_per_structure(bases_single, indices_single; soc=true)
    @test result_single_soc == Dict("C" => 10)
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

    config_indices, _ = Hamster.get_config_index_sample("", conf)
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

    config_indices, _ = Hamster.get_config_index_sample("", conf)
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
@testset "Universal h5 file and filtered config indices" begin
    strc_file = joinpath(path, "structures_universal.h5")  # CsBaBr_3, CsBaCl_3 (10 configs each), CsBaBr_3_PC
    inds_dir = mktempdir()
    inds_file = joinpath(inds_dir, "inds.h5")
    h5open(inds_file, "w") do f
        f["CsBaBr_3/train_config_inds"] = [1, 2, 3]
        f["CsBaBr_3/val_config_inds"] = [4, 5]
        f["CsBaCl_3/train_config_inds"] = collect(1:8)  # no val_config_inds
    end

    # Test 1: read_config_inds from the different sources
    @test Hamster.read_config_inds("", [3, 1], "train_config_inds") == [3, 1]
    @test Hamster.read_config_inds("", "none", "train_config_inds") == Int64[]
    @test Hamster.read_config_inds("", joinpath(path, "config_inds.dat"), "train_config_inds") == [2, 5, 6]
    @test Hamster.read_config_inds("", joinpath(path, "config_inds.h5"), "train_config_inds") == [2, 5, 6]
    @test Hamster.read_config_inds("CsBaBr_3", inds_file, "val_config_inds") == [4, 5]
    @test Hamster.read_config_inds("CsBaCl_3", inds_file, "val_config_inds", missing_ok=true) == Int64[]
    @test_throws Exception Hamster.read_config_inds("CsBaCl_3", inds_file, "val_config_inds")

    # Test 2: Systems are the groups of the h5 file, PC systems only with pc_weight
    conf = get_empty_config()
    set_value!(conf, "XDATCAR", "Supercell", strc_file)
    @test Hamster.get_systems(conf) == ["CsBaBr_3", "CsBaCl_3"]
    set_value!(conf, "pc_weight", "Supercell", 1)
    set_value!(conf, "Nconf", "Supercell", 1)
    @test Hamster.get_systems(conf) == ["CsBaBr_3", "CsBaBr_3_PC", "CsBaCl_3"]

    # Test 3: Filtered sampling excludes train and val indices
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", "1 2 3")
    set_value!(conf, "val_config_inds", "Supercell", "4 5")
    config_indices = Hamster.get_filtered_config_index_sample("", conf, Nconf=3, Nconf_min=1, Nconf_max=10)
    @test length(config_indices) == 3
    @test all(i -> 6 ≤ i ≤ 10, config_indices)
    config_indices = @test_logs (:warn, r"only 5 configurations remain") Hamster.get_filtered_config_index_sample("", conf, Nconf=100, Nconf_min=1, Nconf_max=10)
    @test config_indices == collect(6:10)

    # Test 4: Standard run indices per system with filter_config_inds
    conf = get_empty_config()
    set_value!(conf, "XDATCAR", "Supercell", strc_file)
    set_value!(conf, "Nconf", "Supercell", 4)
    set_value!(conf, "Nconf_max", "Supercell", 100)
    set_value!(conf, "config_inds", "Supercell", inds_file)
    set_value!(conf, "val_config_inds", "Supercell", inds_file)
    set_value!(conf, "filter_config_inds", "Supercell", "true")
    @test Hamster.get_filter_config_inds(conf)
    systems = Hamster.get_systems(conf)
    config_inds, val_inds = @test_logs (:warn, r"CsBaCl_3: only 2") Hamster.get_config_inds_for_systems(systems, MPI.COMM_WORLD, conf, optimize=false)
    @test length(config_inds["CsBaBr_3"]) == 4
    @test all(i -> 6 ≤ i ≤ 10, config_inds["CsBaBr_3"])
    @test config_inds["CsBaCl_3"] == [9, 10]
    @test all(isempty, values(val_inds))

    # Test 5: Without filter_config_inds, the provided train indices are used
    set_value!(conf, "filter_config_inds", "Supercell", "false")
    set_value!(conf, "val_config_inds", "Supercell", "none")
    @test !Hamster.get_filter_config_inds(conf)
    config_inds, _ = Hamster.get_config_inds_for_systems(systems, MPI.COMM_WORLD, conf, optimize=false)
    @test config_inds["CsBaBr_3"][1:3] == [1, 2, 3]
    @test config_inds["CsBaCl_3"][1:8] == collect(1:8)

    # Test 6: Structures in universal mode are built from the system group
    rs_atom, lattice, configs = h5open(strc_file, "r") do f
        read(f["CsBaBr_3/rs_atom"]), read(f["CsBaBr_3/lattice"]), read(f["CsBaBr_3/configs"])
    end
    strcs = get_structures(conf, mode="universal", system="CsBaBr_3", config_indices=[2, 7])
    @test length(strcs) == 2
    rs_ion = Hamster.frac_to_cart(rs_atom, lattice)
    Ts = Hamster.frac_to_cart(Hamster.get_translation_vectors(1), lattice)
    correct_atom_positions = Bool[]
    for (n, index) in enumerate([2, 7]), iion in eachindex(strcs[n].ions)
        push!(correct_atom_positions, strcs[n].ions[iion].pos ≈ rs_ion[:, iion])
        r_ion = strcs[n].ions[iion].pos - strcs[n].ions[iion].dist
        r_config = Hamster.frac_to_cart(configs[:, iion, index], lattice)
        push!(correct_atom_positions, isapprox(minimum(Hamster.normdiff(r_ion, r_config, Ts[:, R]) for R in axes(Ts, 2)), 0, atol=1e-8))
    end
    @test all(correct_atom_positions)
end
