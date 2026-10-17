using Hamster: DOSSettings, compute_dos, cumulative_integral, exact_state_quantities, offsite, projection_label

dos_path = joinpath(@__DIR__, "..", "model", "test_files")
integral(x, y) = cumulative_integral(x, y)[end]

function get_dos_test_config()
    conf = get_empty_config()
    for (key, value) in ["poscar" => joinpath(dos_path, "POSCAR_gaas"), "rcut" => 5, "load_rllm" => true, "verbosity" => 0,
                         "rllm_file" => joinpath(dos_path, "rllm_true.dat"), "init_params" => joinpath(dos_path, "params.dat")]
        set_value!(conf, key, value)
    end
    for ion in ["Ga", "As"]
        set_value!(conf, "orbitals", ion, "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
        set_value!(conf, "n", ion, 3)
    end
    set_value!(conf, "alpha", "Ga", 9); set_value!(conf, "alpha", "As", 13); set_value!(conf, "NNaxes", "As", true)
    return conf
end

dos_test_ks = [0.0 0.25 0.5 0.1; 0.0 0.0 0.5 0.3; 0.0 0.25 0.5 0.2]

@testset "DOS config" begin
    conf = get_empty_config()
    @test Hamster.get_dos(conf) == String[]
    @test Hamster.dos_quantities(conf) == String[]
    set_value!(conf, "dos", "pdos")
    @test Hamster.get_dos(conf) == ["pdos"]
    set_value!(conf, "dos", "pdos cohp")
    @test Hamster.get_dos(conf) == ["pdos", "cohp"]
    @test Hamster.get_dos(conf) isa Vector{String}
    @test Hamster.dos_quantities(conf) == ["pdos", "cohp"]
    set_value!(conf, "dos", "all")
    @test Hamster.dos_quantities(conf) == ["dos", "pdos", "cohp"]
    set_value!(conf, "dos", "bands")
    @test_throws ErrorException Hamster.dos_quantities(conf)
    @test Hamster.get_dos_cohp_pairs(conf) == String[]
    set_value!(conf, "cohp_pairs", "DOS", "Pb-Br Cs-Br")
    @test Hamster.get_dos_cohp_pairs(conf) == ["Pb-Br", "Cs-Br"]
    @test Hamster.get_dos_projection(conf) == "element"
    @test isnan(Hamster.get_dos_nelec(conf))

    # The tags are written to hamster.out at the start of every run
    set_value!(conf, "dos", "pdos cohp")
    mktempdir() do dir
        file = joinpath(dir, "hamster.out")
        Hamster.write_config_tags(conf, filename=file)
        @test occursin("dos = pdos cohp", read(file, String))
        @test occursin("cohp_pairs = Pb-Br Cs-Br", read(file, String))
    end
end

@testset "DOS of a dimer" begin
    # H = [0 t; t 0]: bonding state at t < 0 with COHP contribution 2 × (1/2) × t = t per spin
    t = -1.0
    r = compute_dos([[0.0 t; t 0.0]], zeros(3, 1), ["A-s", "B-s"], [1, 2], DOSSettings(sigma=0.01, nelec=2.0, ne=4000))
    @test r.method == "exact"
    @test r.labels == ["A", "B"] && r.pairs == ["A-B"]
    @test integral(r.energies, r.dos) ≈ 4 atol=1e-5               # 2 states × spin degeneracy 2
    @test integral(r.energies, r.pdos[:, 1]) ≈ 2 atol=1e-5
    @test r.fermi_energy ≈ 0 atol=1e-10
    @test r.icohp_fermi[1] ≈ 2t atol=1e-10                        # 2 electrons in the bonding state
    @test r.icohp[end, 1] ≈ 0 atol=1e-6                           # bonding and antibonding cancel
    @test_throws ErrorException compute_dos([[0.0 t; t 0.0]], zeros(3, 1), ["A-s", "B-s"], [1, 2], DOSSettings(method="tetrahedron"))
end

@testset "DOS sum rules (GaAs)" begin
    conf = get_dos_test_config()
    strc = Structure(conf); basis = Basis(strc, conf)
    ham = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    orbitals, atoms = orbital_labels(strc, basis)
    @test length(orbitals) == 8 && orbitals[1] == "Ga-sp3dr2" && atoms == [1, 1, 1, 1, 2, 2, 2, 2]
    Hr = get_hr(ham, 1)
    r = compute_dos(Hr, ham.Rs[1], orbitals, atoms, DOSSettings(nelec=8.0, ne=4000), ks=dos_test_ks)

    # ∫DOS = spin degeneracy × number of orbitals, ΣPDOS = DOS
    @test integral(r.energies, r.dos) ≈ 16 atol=1e-3
    @test maximum(abs.(vec(sum(r.pdos, dims=2)) .- r.dos)) < 1e-10
    @test r.labels == ["Ga", "As"] && r.pairs == ["Ga-As"]

    # Fermi energy in the gap, ICOHP of the filled valence bands is bonding
    Es, _ = diagonalize(get_hamiltonian(ham, 1, dos_test_ks))
    @test maximum(Es[4, :]) < r.fermi_energy < minimum(Es[5, :])
    @test r.icohp_fermi[1] < 0
    @test r.idos[end] ≈ 16 atol=1e-3

    # Per state: COHP of all pairs + on-site contribution = eigenvalue
    members = [findall(==(l), [findfirst(==(projection_label(o, "element")), r.labels) for o in orbitals]) for l in eachindex(r.labels)]
    E, _, C = exact_state_quantities(Hr, ham.Rs[1], dos_test_ks, atoms, members, [(1, 1), (1, 2), (2, 2)])[2]
    Hk = Matrix(get_hamiltonian(Hr, ham.Rs[1], dos_test_ks[:, 2:2])[1])
    V = eigen(Hermitian(Hk)).vectors
    onsite = real.(diag(V' * (Hk .- offsite(Hk, atoms)) * V))
    @test maximum(abs.(vec(sum(C, dims=1)) .+ onsite .- E)) < 1e-10

    # Orbital projection
    ro = compute_dos(Hr, ham.Rs[1], orbitals, atoms, DOSSettings(projection="orbital"), ks=dos_test_ks)
    @test ro.labels == ["Ga-sp3dr2", "As-sp3dr2"]
    @test maximum(abs.(vec(sum(ro.pdos, dims=2)) .- ro.dos)) < 1e-10

    # Selected COHP pairs
    rp = compute_dos(Hr, ham.Rs[1], orbitals, atoms, DOSSettings(cohp_pairs=["As-Ga"]), ks=dos_test_ks)
    @test rp.pairs == ["Ga-As"]
    @test_throws ErrorException compute_dos(Hr, ham.Rs[1], orbitals, atoms, DOSSettings(cohp_pairs=["Ga-Pb"]), ks=dos_test_ks)

    # SOC: spin orbitals with spin degeneracy 1 give the same DOS for a spin-independent Hamiltonian
    rs = compute_dos(Hamster.apply_spin_basis.(Hr), ham.Rs[1], orbital_labels(strc, basis, soc=true)..., DOSSettings(nelec=8.0, ne=4000),
                     ks=dos_test_ks, soc=true)
    @test rs.labels == ["Ga", "As"]
    @test integral(rs.energies, rs.dos) ≈ 16 atol=1e-3
    @test rs.fermi_energy ≈ r.fermi_energy atol=1e-8
    @test rs.icohp_fermi ≈ r.icohp_fermi atol=1e-8

    # KPM agrees with exact diagonalization
    settings = DOSSettings(nelec=8.0, ne=4000, emin=-14.0, emax=12.0)
    re = compute_dos(Hr, ham.Rs[1], orbitals, atoms, DOSSettings(settings, method="exact"), ks=dos_test_ks)
    rk = compute_dos(Hr, ham.Rs[1], orbitals, atoms, DOSSettings(settings, method="kpm", kpm_moments=256, kpm_vectors=512), ks=dos_test_ks)
    @test rk.method == "kpm"
    @test integral(rk.energies, rk.dos) ≈ 16 rtol=0.01
    ie, ik = cumulative_integral(re.energies, re.dos), cumulative_integral(rk.energies, rk.dos)
    @test maximum(abs.(ie .- ik)) < 0.5                            # different broadening (Gaussian vs. Jackson kernel)
    @test rk.icohp_fermi[1] ≈ re.icohp_fermi[1] rtol=0.05
    @test rk.dos[findfirst(≥(rk.fermi_energy), rk.energies)] < 0.01 * maximum(rk.dos)
end

@testset "DOS post-processing of a standard run" begin
    conf = get_dos_test_config()
    set_value!(conf, "dos", "dos pdos cohp")
    set_value!(conf, "nelec", "DOS", 8)
    set_value!(conf, "ne", "DOS", 500)
    mktempdir() do dir
        cd(dir) do
            Hamster.main(comm, conf, rank=rank, nranks=nranks)
            @test isfile("dos.h5")
            h5open("dos.h5") do f
                @test haskey(f, "1")
                g = f["1"]
                @test length(read(g["energies"])) == 500
                @test sort(keys(g["pdos"])) == ["As", "Ga"]
                @test keys(g["cohp"]) == ["Ga-As"]
                @test haskey(g, "fermi_energy") && haskey(g, "icohp_fermi")
                @test read(attributes(g)["method"]) == "exact"
                @test integral(read(g["energies"]), read(g["dos"])) ≈ 16 atol=0.01
            end
        end
    end
end
