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
@testset "Checks of configuration indices" begin
    calc_path = joinpath(@__DIR__, "..", "calc", "test_files")
    @test_throws "Nconf = 10 configurations are requested" Hamster.get_config_index_sample(Nconf=10, Nconf_max=5, train_mode="md")
    @test_throws "validation set needs 4 configurations" Hamster.get_config_index_sample(Nconf=8, Nconf_max=10, val_ratio=0.5,
                                                                                         validate=true, train_mode="md", val_mode="md")
    conf = get_empty_config()
    set_value!(conf, "config_inds", "Supercell", "inds.txt")
    @test_throws "config_inds = inds.txt: unknown format" Hamster.get_config_index_sample(conf, train_mode="md")
    conf = get_empty_config()
    set_value!(conf, "val_config_inds", "Supercell", "2 3 4")
    _, val_indices = Hamster.get_config_index_sample(conf, Nconf=4, Nconf_max=10, val_ratio=0.25, validate=true,
                                                     train_mode="md", val_mode="md")
    @test val_indices == [2, 3, 4]

    conf = get_empty_config()
    set_value!(conf, "xdatcar", "Supercell", joinpath(calc_path, "XDATCAR"))
    set_value!(conf, "poscar", "Supercell", joinpath(calc_path, "POSCAR_md"))
    @test Hamster.number_of_trajectory_frames(joinpath(calc_path, "XDATCAR")) == 10
    @test Hamster.config_inds_error([1, 10], Int64[], conf, optimize=false) == ""
    @test occursin("configurations 11 are not in", Hamster.config_inds_error([1, 11], Int64[], conf, optimize=false))
    set_value!(conf, "Nconf_max", "Supercell", 20)
    @test occursin("Nconf_max = 20, but", Hamster.config_inds_error([1], Int64[], conf, optimize=false))
    set_value!(conf, "Nconf_max", "Supercell", 10)
    set_value!(conf, "train_mode", "Optimizer", "md")
    set_value!(conf, "train_data", "Optimizer", joinpath(calc_path, "EIGENVAL_gaas"))
    @test occursin("md mode needs an HDF5 file", Hamster.config_inds_error([1], Int64[], conf, optimize=true))
    set_value!(conf, "train_data", "Optimizer", joinpath(calc_path, "eigenval.h5"))
    @test Hamster.config_inds_error([3, 1], Int64[], conf, optimize=true) == ""

    # Structures and eigenvalues follow the order of the indices, also for a list of all configurations
    inds = collect(10:-1:1)
    strcs = get_structures(conf, mode="md", config_indices=collect(1:10))
    strcs_reversed = get_structures(conf, mode="md", config_indices=inds)
    @test all(n -> strcs_reversed[n].ions[1].dist ≈ strcs[inds[n]].ions[1].dist, eachindex(inds))
    data = Hamster.read_eigenvalue_data_from_path(joinpath(calc_path, "eigenval.h5"), collect(1:10), 1:32, 32)
    data_reversed = Hamster.read_eigenvalue_data_from_path(joinpath(calc_path, "eigenval.h5"), inds, 1:32, 32)
    @test all(n -> data_reversed[n].Es ≈ data[inds[n]].Es, eachindex(inds))

    # Configurations are counted (the numbers in the file can be renumbered, e.g. after subsampling)
    mktempdir() do dir
        lines = readlines(joinpath(calc_path, "XDATCAR"))
        iconfig = 0
        renumbered = map(lines) do line
            Hamster.is_configuration_line(line) || return line
            iconfig += 1
            return "Direct configuration=     $(3iconfig)"
        end
        write(joinpath(dir, "XDATCAR"), join(renumbered, "\n") * "\n")
        @test Hamster.number_of_trajectory_frames(joinpath(dir, "XDATCAR")) == 10
        @test Hamster.read_xdatcar(joinpath(dir, "XDATCAR"))[2] ≈ Hamster.read_xdatcar(joinpath(calc_path, "XDATCAR"))[2]
    end
end

@testset "Validation structures from a trajectory of their own" begin
    calc_path = joinpath(@__DIR__, "..", "calc", "test_files")
    mktempdir() do dir
        inds_file = joinpath(dir, "inds.h5")
        h5open(inds_file, "w") do f
            f["train_config_inds"] = [1, 2]
            g = create_group(f, "gaas")
            g["train_config_inds"] = [3]
            g["val_config_inds"] = [4, 5]
        end
        @test Hamster.read_config_inds(inds_file, "config_inds", "train_config_inds", system="gaas") == [3]
        @test Hamster.read_config_inds(inds_file, "config_inds", "train_config_inds", system="cspbbr3") == [1, 2]
        @test Hamster.read_config_inds(inds_file, "config_inds", "train_config_inds") == [1, 2]
        @test Hamster.read_config_inds(inds_file, "val_config_inds", "val_config_inds", system="gaas") == [4, 5]
        @test_throws "and no dataset cspbbr3/val_config_inds" Hamster.read_config_inds(inds_file, "val_config_inds",
                                                                                         "val_config_inds", system="cspbbr3")

        conf = get_empty_config()
        set_value!(conf, "xdatcar", "Supercell", joinpath(calc_path, "XDATCAR"))
        set_value!(conf, "poscar", "Supercell", joinpath(calc_path, "POSCAR_md"))
        set_value!(conf, "Nconf", "Supercell", 10)
        set_value!(conf, "Nconf_max", "Supercell", 10)
        set_value!(conf, "train_mode", "Optimizer", "md")
        set_value!(conf, "train_data", "Optimizer", joinpath(calc_path, "eigenval.h5"))
        set_value!(conf, "val_data", "Optimizer", joinpath(calc_path, "eigenval.h5"))
        set_value!(conf, "val_xdatcar", "Optimizer", joinpath(calc_path, "XDATCAR"))
        @test Hamster.get_validate(conf) && Hamster.separate_val_structures(conf)
        @test Hamster.get_val_poscar(conf) == joinpath(calc_path, "POSCAR_md")
        train, val = Hamster.get_config_inds_for_systems([""], comm, conf, rank=rank)
        @test train[""] == collect(1:10) && val[""] == collect(1:10)
        set_value!(conf, "val_config_inds", "Supercell", "2 4")
        @test Hamster.get_config_inds_for_systems([""], comm, conf, rank=rank)[2][""] == [2, 4]
        set_value!(conf, "val_config_inds", "Supercell", "2 11")
        @test_throws "validation configurations 11 are not in" Hamster.get_config_inds_for_systems([""], comm, conf, rank=rank)
        set_value!(conf, "val_config_inds", "Supercell", "none")
        set_value!(conf, "val_xdatcar", "Optimizer", joinpath(dir, "missing.h5"))
        @test_throws "val_xdatcar = $(joinpath(dir, "missing.h5")): file not found" Hamster.get_config_inds_for_systems([""], comm, conf, rank=rank)
        set_value!(conf, "val_mode", "Optimizer", "pc")
        @test_throws "need val_mode = md or universal" Hamster.get_config_inds_for_systems([""], comm, conf, rank=rank)
    end
end
