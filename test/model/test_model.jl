gaas_poscar = string(@__DIR__) * "/../parse/test_files/POSCAR_gaas"

@testset "TB Model GaAs" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "NNaxes", "As", true)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", string(@__DIR__)*"/test_files/rllm_true.dat")
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "rcut", 5)

    strc = Structure(conf, poscar_path=gaas_poscar)
    basis = Basis(strc, conf)
    model = TBModel(strc, basis, conf, initas=string(@__DIR__)*"/test_files/params.dat")
    Hr = get_hr(model, model.params)
    ks = read_from_file(string(@__DIR__)*"/test_files/kpoints.dat")
    Hk = get_hamiltonian(Hr, strc.Rs, ks)
    Es, vs = diagonalize(Hk)

    Es_correct = read_from_file(string(@__DIR__)*"/test_files/Es.dat")
    @test mean(abs.(Es .- Es_correct)) < 0.002
end

@testset "Test model gradient" begin
    # Test dense implementation
    dL_dHr = [rand(4, 4) for _ in 1:5]
    dL_dHr_t = cat(dL_dHr..., dims=3)
    hs_t = rand(5, 4, 4, 5)
    hs = Matrix{Matrix{Float64}}(undef, 5, 5)
    for v in 1:5, R in 1:5
        hs[v, R] = hs_t[v, :, :, R]
    end
    @tensor dV_t[v] := hs_t[v, i, j, R] * dL_dHr_t[i, j, R]
    dV = Hamster.get_model_gradient(hs, dL_dHr)
    @test dV ≈ dV_t

    # Test sparse implementation
    dL_dHr = [sparse(dL_dHr[i]) for i in 1:5]
    hs = Matrix{SparseMatrixCSC{Float64, Int64}}(undef, 5, 5)
    for v in 1:5, R in 1:5
        hs[v, R] = sparse(hs_t[v, :, :, R])
    end
    dV = Hamster.get_model_gradient(hs, dL_dHr)
    @test dV ≈ dV_t
end

@testset "Test init_params" begin
    # Test 1: Test parameter initialization
    model = TBModel(nothing, zeros(3), zeros(3), [zeros(Int64, 3)], [true, true, true])
    basis = nothing
    Hamster.init_params!(model, initas="ones")
    @test model.params == ones(3)
    @test get_params(model) == ones(3)
    Hamster.init_params!(model, initas="random")
    @test all(0 .< model.params .< 1)

    # Test 2: test setting parameters
    set_params!(model, [1, 2, 3])
    @test get_params(model) == [1, 2, 3]
    @test_throws ErrorException set_params!(model, [1, 2, 3, 4])
end
@testset "TB Model update_tb with several structures" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "orbitals", "As", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
    set_value!(conf, "n", "Ga", 3)
    set_value!(conf, "n", "As", 3)
    set_value!(conf, "alpha", "Ga", 9)
    set_value!(conf, "alpha", "As", 13)
    set_value!(conf, "NNaxes", "As", true)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", string(@__DIR__)*"/test_files/rllm_true.dat")
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "rcut", 5)
    strc = Structure(conf, poscar_path=gaas_poscar)
    basis = Basis(strc, conf)

    # A list of indices (in the order of params.dat) selects these parameters, it is not reduced to all or nothing
    set_value!(conf, "update_tb", "Optimizer", "1 3")
    model = TBModel([strc], [basis], comm, conf, write_summary=false)
    @test model.update == [i ∈ (1, 3) for i in eachindex(model.params)]
    reg = Hamster.Regularization(0., 0., 2)
    dL_dHr = [[ones(size(model.hs[1][1, R])) for R in axes(model.hs[1], 2)]]
    dV = Hamster.get_model_gradient(model, [1], reg, dL_dHr)
    @test all(dV[.!model.update] .== 0)
    @test any(dV[model.update] .≠ 0)

    set_value!(conf, "update_tb", "Optimizer", false)
    @test !any(TBModel([strc], [basis], comm, conf, write_summary=false).update)
    set_value!(conf, "update_tb", "Optimizer", true)
    @test all(TBModel([strc], [basis], comm, conf, write_summary=false).update)
    @test_throws ErrorException TBModel([strc], [basis], comm, conf, update_tb=[true, false], write_summary=false)

    # Indices beyond the number of parameters are an error
    set_value!(conf, "update_tb", "Optimizer", "1 1000")
    @test_throws ErrorException TBModel([strc], [basis], comm, conf, write_summary=false)
end

@testset "Test init_params keywords and files" begin
    @test Hamster.get_init_mode("ones") == :ones
    @test Hamster.get_init_mode("Zeros") == :zeros
    @test Hamster.get_init_mode("rand") == :rand
    @test Hamster.get_init_mode("random") == :rand
    @test Hamster.get_init_mode("z") == :zeros
    @test_throws ErrorException Hamster.get_init_mode("no_such_file.dat")

    # A parameter file whose name starts with o, r or z is read as a file
    labels = Hamster.string_to_param_label.(["NN1_Ga+Ga_ssσ", "NN1_As+Ga_ssσ", "NN1_As+As_ssσ"])
    mktempdir() do dir
        for name in ["old_params.dat", "results_params.dat", "zahn_params.dat"]
            file = joinpath(dir, name)
            write(file, "NN1_Ga+Ga_ssσ  2.5\nNN1_As+Ga_ssσ  -1.5\n")
            @test Hamster.get_init_mode(file) == :file
            model = TBModel(nothing, ones(3), labels, [collect(1:3)], fill(true, 3))   # as in the TBModel constructors
            # The parameter that is missing in the file keeps the value 1 and is reported
            @test_logs (:warn, r"1 of 3 TB parameters are not in") Hamster.init_params!(model, initas=file)
            @test model.params == [2.5, -1.5, 1.0]
            @test_logs Hamster.init_params!(model, initas=file, warn=false)
        end
    end
end

@testset "Radial exponents" begin
    files = joinpath(@__DIR__, "test_files")
    function radial_conf(radial; update=true)
        local rc = get_empty_config()
        for (element, orbitals) in [("Cs", "s"), ("Pb", "s px py pz"), ("Br", "px py pz")]; set_value!(rc, "orbitals", element, orbitals); end
        for (key, value) in ["poscar" => joinpath(files, "POSCAR_cspbbr3_distorted"), "load_rllm" => true, "rcut" => 5.5,
                             "rllm_file" => joinpath(files, "rllm_cspbbr3.dat"), "verbosity" => 0, "radial_exponent" => radial]
            set_value!(rc, key, value)
        end
        set_value!(rc, "update_radial_exponent", "Optimizer", update)
        return rc
    end
    conf = radial_conf(true)
    strc = Structure(conf); basis = Basis(strc, conf)
    plain = TBModel([strc], [basis], comm, radial_conf(false), write_summary=false)
    model = TBModel([strc], [basis], comm, conf, write_summary=false)
    @test get_hr(model, Hamster.Dense(), 1) == get_hr(plain, Hamster.Dense(), 1)
    hop = Hamster.hopping_parameters(model)
    @test !isempty(hop) && length(hop) < length(model.params) && length(get_params(model)) == 2length(model.params)

    h, logr = Hamster.get_geometry_tensor(strc, basis, conf, log_distances=true)
    pos = Hamster.get_ion_positions(strc.ions, apply_distortion=true); Ts = Hamster.frac_to_cart(strc.Rs, strc.lattice)
    atom = reduce(vcat, [fill(i, n) for (i, n) in enumerate(size(basis))])
    for v in hop[1:3], R in axes(h, 2)
        rows, cols, _ = findnz(h[v, R])
        @test exp.(logr[v, R]) ≈ [norm(pos[atom[i]] - (pos[atom[j]] + Ts[:, R])) for (i, j) in zip(rows, cols)]
    end
    @test all(iszero, logr[findfirst(p -> p.nnlabel == 0, model.param_labels), Hamster.findR0(strc.Rs)])

    c = [v ∈ hop ? 0.5 + 0.3 * sin(v) : 0.0 for v in eachindex(model.params)]
    θ0 = vcat(1 .+ 0.1 .* sin.(eachindex(model.params)), c)
    Hamster.set_params!(model, θ0)
    for v in hop, R in axes(h, 2)
        @test nonzeros(model.hs[1][v, R]) ≈ nonzeros(h[v, R]) .* exp.(-c[v] .* logr[v, R])
    end
    @test Hamster.base_geometry_tensor(model, 1) == h

    Hr = get_hr(model, Hamster.Dense(), 1)
    G = [[sin(i + 2j + 3R) for i in axes(Hr[R], 1), j in axes(Hr[R], 2)] for R in eachindex(Hr)]
    loss(θ) = (Hamster.set_params!(model, θ); sum(sum(G[R] .* real(get_hr(model, Hamster.Dense(), 1)[R])) for R in eachindex(G)))
    reg = Hamster.Regularization(0., 0., 2)
    grad = Hamster.get_model_gradient(model, [1], reg, [G])
    checked = vcat(1:3, length(model.params) .+ hop[1:4])
    fd = [(loss(θ0 .+ 1e-6 .* (eachindex(θ0) .== p)) - loss(θ0 .- 1e-6 .* (eachindex(θ0) .== p))) / 2e-6 for p in checked]
    Hamster.set_params!(model, θ0)
    @test grad[checked] ≈ fd rtol=1e-6
    @test all(iszero, grad[length(model.params) .+ setdiff(eachindex(model.params), hop)])

    terms = Hamster.linear_terms(model, 1); p = Hamster.device_param_values(model)
    hr = [zeros(size(G[1])) for _ in G]; raw = zeros(length(p))
    for (key, col, val) in zip(terms.terms.keys, terms.terms.cols, terms.terms.vals)
        hr[key[1]][key[2], key[3]] += val * p[col]; raw[col] += val * G[key[1]][key[2], key[3]]
    end
    @test all(hr[R] ≈ real(Hr[R]) for R in eachindex(hr))
    @test Hamster.get_model_gradient(model, [1], reg, Hamster.RawModelGradient(raw)) ≈ grad

    fixed = TBModel([strc], [basis], comm, radial_conf(true, update=false), write_summary=false)
    Hamster.set_params!(fixed, θ0)
    @test all(iszero, Hamster.get_model_gradient(fixed, [1], reg, [G])[length(model.params)+1:end])
    Hamster.copy_params!(fixed, model)
    @test get_hr(fixed, Hamster.Dense(), 1) == Hr

    mktempdir() do dir
        cd(dir) do
            Hamster.write_params(model, conf)
            @test count("exponent_", read("params.dat", String)) == length(hop)
            labels, values = Hamster.read_params("params.dat")[10:11]
            @test labels == model.param_labels[hop] && values == c[hop]
            set_value!(conf, "init_params", "params.dat")
            loaded = TBModel([strc], [basis], comm, conf, write_summary=false)
            @test get_params(loaded) ≈ θ0 && get_hr(loaded, Hamster.Dense(), 1) ≈ Hr
            @test_throws ErrorException TBModel(strc, basis, conf)
            set_value!(conf, "radial_exponent", false)
            @test get_params(TBModel([strc], [basis], comm, conf, write_summary=false)) ≈ θ0[1:length(model.params)]
        end
    end
end
