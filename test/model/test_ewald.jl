function brute_force_periodic_coulomb(
    pos,
    q,
    box;
    nimg::Int=3
)
    N = length(pos)

    phi = zeros(Float64, N)
    E = 0.0
    @views for i in 1:N
        ri = pos[i]

        for j in 1:N
            rj = pos[j]
            qj = q[j]

            for nx in -nimg:nimg, ny in -nimg:nimg, nz in -nimg:nimg
                if i == j && nx == 0 && ny == 0 && nz == 0
                    continue
                end

                shift = box * [nx, ny, nz]
                dr = ri - (rj + shift)
                r = norm(dr)

                phi[i] += Hamster.ke * qj / r
            end
        end
    end

    E = 0.5 * sum(q .* phi)

    return (
        potentials = phi,
        energy = E,
        nimg = nimg,
    )
end

@testset "Ewald sum" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "s")
    set_value!(conf, "orbitals", "Pb", "s px py pz")
    set_value!(conf, "orbitals", "Br", "px py pz")
    set_value!(conf, "alpha", "Cs", 9)
    set_value!(conf, "alpha", "Pb", 13)
    set_value!(conf, "alpha", "Br", 13)
    set_value!(conf, "poscar", joinpath(@__DIR__, "test_files/POSCAR_cspbbr3"))
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", joinpath(@__DIR__, "test_files/rllm_cspbbr3.dat"))
    set_value!(conf, "verbosity", 2)
    set_value!(conf, "rcut", 5.5)

    strc = Structure(conf)

    poscar = Hamster.read_poscar(Hamster.get_poscar(conf))
    rs_atom = Hamster.frac_to_cart(poscar.rs_atom, poscar.lattice)
    pos = Vector{Float64}[r for r in eachcol(rs_atom)]
    charges = Dict{String, Float64}("Pb"=>2, "Cs"=>1, "Br"=>-1)
    atom_types = Hamster.number_to_element.(poscar.atom_types)
    q = [charges[type] for type in atom_types]

    box = poscar.lattice
    ewald = Hamster.ewald_sum(pos, q, box, Rs=strc.Rs, point_grid=strc.point_grid, rcut=5.5)

    # Test 1: test Ewald energy vs reference implementation
    latvecs = [v for v in eachcol(box)]
    sys = Ewalder.System(; latvecs=latvecs, pos=pos)
    E = Ewalder.energy(sys; charges=q)
    @test isapprox(ewald.energy, E * Hamster.ke, rtol=1e-3)

    # Test 2: test Ewald energy and potential vs larger cut-off real-space only Ewald
    ewald2 = Hamster.ewald_sum(pos, q, box, rcut=20, include_long=false)
    @test isapprox(ewald.energy, ewald2.energy, rtol=1e-3)
    @test all(@. isapprox(ewald.potentials, ewald2.potentials, rtol=1e-3))

    # Test 3: test Ewald energy and potential vs brute force periodic coulomb
    ewald3 = brute_force_periodic_coulomb(pos, q, box; nimg=7)
    @test isapprox(ewald.energy, ewald3.energy, rtol=1e-3)
    @test all(@. isapprox(ewald.potentials, ewald3.potentials, rtol=1e-2))

    # Test 4: test Zahn/Wolf potentials
    zahn = Hamster.ewald_sum(pos, q, box, rcut=25, alpha=0.15, method="zahn")
    wolf = Hamster.ewald_sum(pos, q, box, rcut=25, alpha=0.15, method="wolf")

    @test isapprox(zahn.energy, wolf.energy, rtol=1e-3)
    @test all(@. isapprox(zahn.potentials, wolf.potentials, rtol=1e-2))

    # Test 5: real-space sum (translation vectors from the bounding-sphere estimate, threaded loop over grid cells)
    # versus a direct double sum over atoms and the translation vectors of the pair search
    for (method, rc, α) in [("ewald", 5.5, 0.0), ("wolf", 12.0, 0.2), ("zahn", 12.0, 0.2)]
        Rs_pairs = Hamster.get_translation_vectors(hcat(pos...), box; Rmax=Hamster.get_Rmax(box, rcut=rc), rcut=rc)
        Rs = Hamster.ewald_translation_vectors(pos, box, rc)
        @test all(R -> R in eachcol(Rs), eachcol(Rs_pairs))           # superset of the translations with neighbors

        res = Hamster.ewald_sum(pos, q, box, rcut=rc, method=method, alpha=α, include_long=false, subtract_self=false)
        kernel = Hamster.get_realspace_kernel(method)
        Ts = Hamster.frac_to_cart(Rs_pairs, box)
        φ_direct = zeros(length(pos))
        for i in eachindex(pos), j in eachindex(pos), R in axes(Ts, 2)
            r = norm(pos[i] - pos[j] - Ts[:, R])
            1e-14 < r < rc && (φ_direct[i] += q[j] * kernel(r, res.alpha, rc))
        end
        @test res.potentials ≈ Hamster.ke .* φ_direct atol=1e-10
    end
end

@testset "Ewald Onsites" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "s")
    set_value!(conf, "orbitals", "Pb", "s px py pz")
    set_value!(conf, "orbitals", "Br", "px py pz")
    set_value!(conf, "qeff", "Cs", 1)
    set_value!(conf, "qeff", "Pb", 2)
    set_value!(conf, "qeff", "Br", -1)
    set_value!(conf, "poscar", joinpath(@__DIR__, "test_files/POSCAR_cspbbr3_distorted"))
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", joinpath(@__DIR__, "test_files/rllm_cspbbr3.dat"))
    set_value!(conf, "verbosity", 2)
    set_value!(conf, "rcut", 5.5)

    strc = Structure(conf)
    basis = Basis(strc, conf)
    zahn_model = EwaldOnsites([strc], [basis], comm, conf, method="zahn", rcut=35)
    ewald_model = EwaldOnsites([strc], [basis], comm, conf)

    @test all(@. isapprox(zahn_model.potentials, ewald_model.potentials, rtol=1e-2))
    Hr = get_hr(ewald_model, Hamster.Dense(), 1, apply_soc=false)

    @test size(Hr[1])[1] == length(basis)

    s = Hamster.correction_statistics([-2.0, -1.0, 0.0, 1.0, 2.0])
    @test s.n == 5 && s.σ ≈ √2 && s.min == -2 && s.max == 2
    @test s.tail ≈ 2 / √2 && s.frac3σ == 0
    @test isnan(Hamster.correction_statistics(Float64[]).σ)

    mktempdir() do dir
        cd(dir) do
            EwaldOnsites([strc], [basis], comm, conf, label="no file")
            @test !isfile("hamster.out")

            touch("hamster.out")
            EwaldOnsites([strc], [basis], comm, conf, label="test set")
            EwaldOnsites([strc], [basis], comm, conf, label="skipped", write_summary=false)
            EwaldOnsites([strc], [basis], comm, conf, rcut=8.0, label="large rcut")
            out = read("hamster.out", String)
            @test occursin("Ewald Model Summary (test set)", out)
            @test !occursin("skipped", out)
            for element in ["Cs", "Pb", "Br"]
                @test occursin(Regex("\\n    $element\\s+\\d+ "), out)
            end
            @test count("real-space neighbor grid:    rebuilt", out) == 1
        end
    end
end
@testset "Ewald coupling gradient and parameter file" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "s")
    set_value!(conf, "orbitals", "Pb", "s px py pz")
    set_value!(conf, "orbitals", "Br", "px py pz")
    set_value!(conf, "qeff", "Cs", 1)
    set_value!(conf, "qeff", "Pb", 2)
    set_value!(conf, "qeff", "Br", -1)
    set_value!(conf, "poscar", joinpath(@__DIR__, "test_files/POSCAR_cspbbr3_distorted"))
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", joinpath(@__DIR__, "test_files/rllm_cspbbr3.dat"))
    set_value!(conf, "rcut", 5.5)
    set_value!(conf, "verbosity", 0)
    strc = Structure(conf)
    basis = Basis(strc, conf)
    model = EwaldOnsites([strc], [basis], comm, conf, update=true)
    model.params .= [0.4, 0.6, 0.8][1:length(model.params)]
    reg = Hamster.Regularization(conf, lambda=0.0)

    # dL/dγ for the linear loss L = Re Σ_R Σ_ij G_R,ij Hr_R,ij versus central finite differences
    for soc in (false, true)
        Hr = get_hr(model, Hamster.Dense(), 1, apply_soc=soc)
        G = [randn(ComplexF64, size(H)) for H in Hr]
        loss(γ) = (model.params .= γ; real(sum(sum(G[R] .* get_hr(model, Hamster.Dense(), 1, apply_soc=soc)[R]) for R in eachindex(G))))
        γ0 = copy(model.params)
        grad = Hamster.get_model_gradient(model, [1], reg, [G]; soc=soc)
        fd = [(loss(γ0 .+ 1e-4 .* (1:length(γ0) .== p)) - loss(γ0 .- 1e-4 .* (1:length(γ0) .== p))) / 2e-4 for p in eachindex(γ0)]
        model.params .= γ0
        @test grad ≈ fd rtol=1e-6
    end

    # Without update, no gradient
    model_fixed = EwaldOnsites([strc], [basis], comm, conf)
    @test all(iszero, Hamster.get_model_gradient(model_fixed, [1], reg, [get_hr(model_fixed, Hamster.Dense(), 1)]; soc=false))

    # Coupling per species from the atom blocks (e.g. for a hyperparameter search over Pb_charge_scale)
    set_value!(conf, "charge_scale", "Ewald", 0.5)
    set_value!(conf, "charge_scale", "Pb", 0.3)
    per_species = EwaldOnsites([strc], [basis], comm, conf)
    for (label, γ) in zip(per_species.param_labels, per_species.params)
        @test γ ≈ (Hamster.number_to_element(label) == "Pb" ? 0.3 : 0.5)
    end

    # Fitted coupling written to params.dat (after the TB and SOC parameters, which are kept) and read with init_params
    mktempdir() do dir
        cd(dir) do
            Hamster.write_params(Hamster.ParameterLabel[], Float64[], ["Pb"], [0.25], conf)   # params.dat as from TB and SOC
            Hamster.write_params(model, conf)
            _, _, soc_types, soc_params, _, ewald_types, ewald_params = Hamster.read_params("params.dat")
            @test soc_types == ["Pb"] && soc_params == [0.25]
            @test ewald_params ≈ model.params
            set_value!(conf, "init_params", "Ewald", joinpath(dir, "params.dat"))
            loaded = EwaldOnsites([strc], [basis], comm, conf)
            @test loaded.params ≈ model.params
            @test loaded.param_labels == model.param_labels
        end
    end
end

@testset "Ewald settings in the parameter file" begin
    @test Hamster.ewald_alpha("ewald", 5.5, 0) ≈ 4 / 5.5
    @test Hamster.ewald_alpha("wolf", 15.0, 0) ≈ 0.2
    @test Hamster.ewald_alpha("zahn", 15.0, 0.3) == 0.3

    function cspbbr3_conf()
        conf = get_empty_config()
        set_value!(conf, "orbitals", "Cs", "s")
        set_value!(conf, "orbitals", "Pb", "s px py pz")
        set_value!(conf, "orbitals", "Br", "px py pz")
        set_value!(conf, "qeff", "Cs", 1)
        set_value!(conf, "qeff", "Pb", 2)
        set_value!(conf, "qeff", "Br", -1)
        set_value!(conf, "poscar", joinpath(@__DIR__, "test_files/POSCAR_cspbbr3_distorted"))
        set_value!(conf, "load_rllm", true)
        set_value!(conf, "rllm_file", joinpath(@__DIR__, "test_files/rllm_cspbbr3.dat"))
        set_value!(conf, "rcut", 5.5)
        set_value!(conf, "verbosity", 0)
        return conf
    end
    conf = cspbbr3_conf()
    set_value!(conf, "method", "Ewald", "wolf")
    set_value!(conf, "rcut", "Ewald", 18.0)
    strc = Structure(conf)
    basis = Basis(strc, conf)
    model = EwaldOnsites([strc], [basis], comm, conf)

    mktempdir() do dir
        cd(dir) do
            Hamster.write_params(Hamster.ParameterLabel[], Float64[], ["Pb"], [0.25], conf)
            Hamster.write_params(model, conf)
            header = Hamster.read_params("params.dat")[5]
            @test header["ewald_method"] == "wolf"
            @test parse(Float64, header["ewald_rcut"]) == 18.0
            @test parse(Float64, header["ewald_alpha"]) ≈ 3 / 18
            @test parse(Float64, header["ewald_qeff_Pb"]) == 2.0 && parse(Float64, header["ewald_qeff_Br"]) == -1.0
            @test Hamster.check_consistency(header, conf)

            # Not set in the config: taken from the file
            new = cspbbr3_conf()
            set_value!(new, "init_params", "Ewald", "params.dat")
            Hamster.apply_ewald_settings!(new)
            @test Hamster.get_ewald_method(new) == "wolf"
            @test Hamster.get_ewald_rcut(new) == 18.0 && Hamster.get_ewald_alpha(new) ≈ 3 / 18
            @test EwaldOnsites([strc], [basis], comm, new).potentials ≈ model.potentials

            # Set in the config and different: error, unless check_params = false
            different = cspbbr3_conf()
            set_value!(different, "init_params", "Ewald", "params.dat")
            set_value!(different, "rcut", "Ewald", 15.0)
            @test_throws ErrorException Hamster.apply_ewald_settings!(different)
            set_value!(different, "qeff", "Pb", 3)
            set_value!(different, "check_params", "Ewald", false)
            Hamster.apply_ewald_settings!(different)
            @test Hamster.get_ewald_rcut(different) == 15.0 && Hamster.get_qeff(different, "Pb") == 3.0

            # Same settings in the config: no error
            same = cspbbr3_conf()
            set_value!(same, "init_params", "Ewald", "params.dat")
            set_value!(same, "method", "Ewald", "wolf")
            set_value!(same, "rcut", "Ewald", 18)
            Hamster.apply_ewald_settings!(same)
            @test Hamster.get_ewald_alpha(same) ≈ 3 / 18

            # Ewald: rcut and alpha only affect the numerics and are not taken from the file
            ewald_conf = cspbbr3_conf()
            ewald = EwaldOnsites([strc], [basis], comm, ewald_conf)
            Hamster.write_params(ewald, ewald_conf)
            other = cspbbr3_conf()
            set_value!(other, "init_params", "Ewald", "params.dat")
            set_value!(other, "rcut", "Ewald", 8.0)
            Hamster.apply_ewald_settings!(other)
            @test Hamster.get_ewald_rcut(other) == 8.0
        end
    end
end
