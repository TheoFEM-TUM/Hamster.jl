path = string(@__DIR__) * "/test_files/"

@testset "EffectiveHamiltonian for PC" begin
    conf = get_empty_config()
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "NNaxes", "As", true)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", joinpath(path, "rllm_true.dat"))
    set_value!(conf, "rcut", 5)
    set_value!(conf, "init_params", joinpath(path, "params.dat"))
    set_value!(conf, "verbosity", 0)

    # Test that effective Hamiltonian model gives correct eigenvalues
    strc = Structure(conf); basis = Basis(strc, conf)
    eff_ham = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    ks = read_from_file(joinpath(path, "kpoints.dat"))
    Es_correct = read_from_file(joinpath(path, "Es.dat"))
    Hk = get_hamiltonian(eff_ham, 1, ks)
    Es, vs = diagonalize(Hk)
    @test mean(abs.(Es .- Es_correct)) < 0.01

    # Test empty model
    eff_empty = EffectiveHamiltonian([], [], comm, conf, rank=rank, nranks=nranks)
    @test eff_empty.Nstrc == 0
    @test length(eff_empty.models) == 0
    @test eltype(eff_empty.models) <: Nothing
end
@testset "copy_params! matches Ewald and SOC parameters by element" begin
    # The training and validation Hamiltonians can list their elements in different orders (they are built from
    # structures distributed differently over the ranks), so the parameters must be copied by element, not by position
    ewald(labels, params) = Hamster.EwaldOnsites(params, UInt8.(labels), Vector{Vector{Float64}}(), Vector{Vector{UInt8}}(),
                                                 Vector{Vector{Int64}}(), zeros(Int64, 2, 0), true)
    soc(labels, params) = Hamster.SOCModel(params, UInt8.(labels), Vector{Vector{UInt8}}(), Hamster.OrderedDict{UInt8, Matrix{Float64}}(),
                                           zeros(Int64, 2, 0), true)
    eff(models) = Hamster.EffectiveHamiltonian(0, models, String[], nothing, nothing, 0.0, nothing, false, Matrix{Float64}[], nothing)

    br, ba, pb = 35, 56, 82
    train = eff((ewald([br, ba], [1.1, 2.2]), soc([br, ba, pb], [0.1, 0.2, 0.3])))
    val = eff((ewald([ba, br], [0.0, 0.0]), soc([pb, br], [0.0, 0.0])))
    copy_params!(val, train)
    @test val.models[1].params == [2.2, 1.1]
    @test val.models[2].params == [0.3, 0.1]

    # The receiving models get copies of the values (not the arrays of the sending models)
    train.models[1].params .= 0.0
    @test val.models[1].params == [2.2, 1.1]
end
