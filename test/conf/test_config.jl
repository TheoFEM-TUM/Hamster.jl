@testset "Config" begin
    conf = Hamster.get_empty_config()

    # Test convert value
    convert_value = Hamster.convert_value
    @test convert_value("7.0") == 7.0
    @test convert_value("1 2 3") == [1, 2, 3]
    @test convert_value("1.0 2 3") == Float64[1, 2, 3]
    @test convert_value("true") == true
    @test convert_value("false") == false
    @test convert_value("True") == true
    @test convert_value("FALSE") == false
    @test convert_value("1 1 1; 2 2 2") == [1 1 1; 2 2 2]
    @test convert_value("1 1 1; 2 2 2.0") == Float64[1 1 1; 2 2 2]

    # Parameter that is not set should return "default"
    @test conf("rcut") == "default"

    # Test setting parameter values
    set_value!(conf, "rcut", "7.0")
    @test conf("rcut") == 7.0

    # Value in non-existing block should be default
    @test conf("lr", "Optimizer") == "default"

    # Create block and set value in block
    set_value!(conf, "lr", "Optimizer", "0.1")
    @test conf("lr", "Optimizer") == 0.1

    # Lookup multiple keys
    @test conf(["rcut", "interpolate_rllm"]) == [7.0, "default"]

    # Keys should no be case sensitive
    set_value!(conf, "POSCAR", "POSCAR_gaas")
    @test conf("poscar") == "POSCAR_gaas"
    @test conf("POSCAR") == conf("poscar")

    set_value!(conf, "poScAr", "Supercell", "SC_POSCAR")
    @test conf("poscar", "Supercell") == "SC_POSCAR"
    @test conf("POSCAR", "Supercell") == "SC_POSCAR"
end

import Hamster.@configtag, Hamster.ConfigTag, Hamster.CONFIG_TAGS, Hamster.get_tag

@testset "Config tags" begin
    conf = get_empty_config()
    @configtag some_float Float64 1e-5 "This is a tag of type Float64."
    float_tag = Hamster.ConfigTag{Float64}("some_float", _ -> 1e-5,
        "This is a tag of type Float64.")
    @test get_tag(conf, float_tag) == 1e-5
    set_value!(conf, "some_float", 2.1)
    @test get_tag(conf, float_tag) == 2.1
    @test get_some_float(conf) == 2.1
    test_tag = CONFIG_TAGS[end]
    @test test_tag.name == float_tag.name
    @test test_tag.description == float_tag.description
    @test test_tag.block == float_tag.block

    vector_tag = ConfigTag{Vector{Int64}}("some_vector", conf->[1, 2],
        "This is a tag of type Vector{Int64}.")
    @configtag vector_tag Vector{Float64} [1.] "This is a tag of type Vector{Float64}."
    set_value!(conf, "vector_tag", 1e-5)

    @test get_vector_tag(conf) == [1e-5]

    # Options of the macro: readable default, category, see also, custom getter
    @configtag some_computed Float64 2 * get_some_float(conf) "Computed default." "ML" default_doc="twice some_float" category="model" see_also=["some_float"]
    tag = CONFIG_TAGS[end]
    @test tag.block == "ML" && tag.default_doc == "twice some_float" && tag.category == "model" && tag.see_also == ["some_float"]
    @test get_ml_some_computed(conf) ≈ 2 * 2.1
    get_some_custom(conf) = 7
    @configtag some_custom Int64 0 "Custom getter." getter=:custom
    @test get_tag(conf, CONFIG_TAGS[end]) == 7 && CONFIG_TAGS[end].default_doc == "0"
    @test Hamster.literal_default_doc(:(String[])) == "(empty)" && Hamster.literal_default_doc(:([1, 2])) == "1 2"
    @test Hamster.literal_default_doc(:(get_rcut(conf))) == ""

    # Remove the test tags again
    filter!(t -> t.name ∉ ("some_float", "vector_tag", "some_computed", "some_custom"), CONFIG_TAGS)
end

@testset "Registry of input tags" begin
    tags = Hamster.config_tags()
    blocks = Dict(b[1] => b[3] for b in Hamster.CONFIG_BLOCKS)
    @test length(tags) > 100
    @test all(t -> !isempty(strip(t.description)), tags)
    @test all(t -> !isempty(t.default_doc), tags)                        # computed defaults need a default_doc
    @test all(t -> haskey(blocks, t.block) && t.category ∈ blocks[t.block], tags)
    @test allunique([(t.block, lowercase(t.name)) for t in tags])
    @test all(t -> all(name -> !isempty(Hamster.find_tags(name)), t.see_also), tags)
    @test all(t -> t.per_atom == (t.block == "atom"), tags)

    # Lookup and suggestions
    @test [t.block for t in Hamster.find_tags("rcut")] == ["Options", "ML", "Ewald"]
    @test Hamster.suggest_tags("ncluter")[1].name == "ncluster"
    @test Hamster.suggest_tags("rcutt", block="Options")[1].block == "Options"
    @test Hamster.string_distance("abc", "acb") == 1

    # Help, list and command line
    help = sprint(Hamster.tag_help, "init_params")
    @test occursin("Options block", help) && occursin("ML block", help) && occursin("SOC block", help)
    @test occursin("Did you mean: ncluster", sprint(Hamster.tag_help, "ncluter"))
    @test occursin("npoints", sprint(Hamster.list_tags, "ML")) && !occursin("rcut_tol", sprint(Hamster.list_tags, "ML"))
    @test occursin("charge_scale", sprint(Hamster.list_tags, "ewald"))
    @test occursin("No input tags", sprint(Hamster.list_tags, "zzzz"))
    @test strip(@capture_out(Hamster.tag_cli(["--tag", "neig"]))) |> s -> startswith(s, "neig")

    # Check of a config: wrong block, typos and unknown blocks; valid entries pass
    conf = get_empty_config()
    set_value!(conf, "xdatcar", "x.h5"); set_value!(conf, "rcutt", 5)
    set_value!(conf, "nconf", "Supercell", 3); set_value!(conf, "wE_10-20", "Optimizer", 0.5); set_value!(conf, "wk_1", "Optimizer", 5)
    set_value!(conf, "alpha", "Pb", 37); set_value!(conf, "qef", "Pb", 1); set_value!(conf, "x", "Foo", 1)
    messages = Hamster.check_config_tags(conf)
    @test length(messages) == 4
    @test any(m -> occursin("\"xdatcar\"", m) && occursin("Supercell block", m), messages)
    @test any(m -> occursin("\"rcutt\"", m) && occursin("did you mean rcut", m), messages)
    @test any(m -> occursin("\"qef\"", m) && occursin("qeff", m), messages)
    @test any(m -> occursin("Unknown block \"Foo\"", m), messages)

    # Warnings when reading a config file
    mktempdir() do dir
        file = joinpath(dir, "hconf")
        write(file, "begin Options\n  xdatcar = traj.h5\n  rcut = 5\nend\nbegin ML\n  npoints = 10\nend\n")
        @test_logs (:warn, r"belongs in the Supercell block") get_config(filename=file)
    end

    # Generated documentation: one page per block and an index, every tag once with its anchor
    mktempdir() do dir
        files = Hamster.write_tag_docs(dir)
        @test files[1] == "tags_index.md" && "tags_options.md" ∈ files && "tags_atom.md" ∈ files
        pages = join([read(joinpath(dir, f), String) for f in files[2:end]])
        index = read(joinpath(dir, "tags_index.md"), String)
        @test all(t -> count("(@id $(Hamster.anchor(t)))", pages) == 1 && occursin("(@ref $(Hamster.anchor(t)))", index), tags)
        @test occursin("(@id options-block)", read(joinpath(dir, "tags_options.md"), String))
        @test occursin("(@id block-labels)", index)
    end
end