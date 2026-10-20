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

@testset "KPM moments" begin
    rng = Hamster.Random.MersenneTwister(5)
    N, M, Nvec = 40, 12, 3
    A = sprandn(rng, ComplexF64, N, N, 0.2)
    H = (A + A') / 2
    H̃ = H ./ (1.01 * maximum(sum(abs.(H), dims=2)))
    Hoff = Hamster.offsite(H, repeat(1:10, inner=4))
    X = randn(rng, ComplexF64, 3, N)
    @test Hamster.hermitian_mul!(similar(X), H, X) ≈ transpose(H * transpose(X))

    members = [collect(1:2:N), collect(2:2:N)]
    pairs = [(1, 1), (1, 2)]
    μ_dos, μ_pdos, μ_cohp = Hamster.kpm_moments(H̃, Hoff, members, pairs, M, Nvec)
    # Dense Chebyshev recursion with the same random-phase vectors
    ref_dos, ref_pdos, ref_cohp = zeros(M), zeros(M, 2), zeros(M, 2)
    for ivec in 1:Nvec
        v = exp.(2π * im .* rand(Hamster.Random.MersenneTwister(1234 + ivec), N))
        T = [v, Matrix(H̃) * v]
        for m in 3:M
            push!(T, 2 * Matrix(H̃) * T[end] - T[end-1])
        end
        for m in 1:M
            ref_dos[m] += real(dot(v, T[m])) / Nvec
            for (l, B) in enumerate(members)
                ref_pdos[m, l] += real(dot(v[B], T[m][B])) / Nvec
            end
            for (p, (a, b)) in enumerate(pairs)
                x = zeros(ComplexF64, N); x[members[a]] = v[members[a]]
                ref_cohp[m, p] += (a == b ? 1 : 2) * real(dot((Matrix(Hoff) * x)[members[b]], T[m][members[b]])) / Nvec
            end
        end
    end
    @test μ_dos ≈ ref_dos
    @test μ_pdos ≈ ref_pdos
    @test μ_cohp ≈ ref_cohp

    # Exact moments of a local label from unit vectors (in batches): diagonal of Tₘ, and the COHP of the pairs with it
    Tm = [Matrix{ComplexF64}(I, N, N), Matrix(H̃)]
    for m in 3:M; push!(Tm, 2 * Matrix(H̃) * Tm[end] - Tm[end-1]); end
    for local_label in (1, 2)
        μ_l, local_pairs, μ_c = Hamster.kpm_local_moments(H̃, Hoff, members, pairs, [local_label], M, batch=7)
        @test μ_l[:, 1] ≈ [real(sum(diag(Tm[m])[members[local_label]])) for m in 1:M]
        @test local_pairs == (local_label == 1 ? [1, 2] : [2])
        for (n, p) in enumerate(local_pairs)
            a, b = pairs[p]
            ref = [(a == b ? 1 : 2) * real(tr(Matrix(Hoff)[members[a], members[b]] * Tm[m][members[b], members[a]])) for m in 1:M]
            @test μ_c[:, n] ≈ ref
        end
    end
end

@testset "Defect projection" begin
    pc = Hamster.read_poscar(joinpath(dos_path, "POSCAR_gaas"))
    sites(atom) = [(pc.rs_atom[:, atom] .+ [n1, n2, n3]) ./ 2 for n1 in 0:1 for n2 in 0:1 for n3 in 0:1]
    mktempdir() do dir
        # 2x2x2 GaAs supercell with an As antisite (As on the first Ga site)
        poscar = joinpath(dir, "POSCAR_antisite")
        open(poscar, "w") do f
            println(f, "GaAs antisite\n1.0"); for a in eachcol(pc.lattice); println(f, join(2 .* a, " ")); end
            println(f, "Ga As\n7 9\nDirect"); foreach(r -> println(f, join(r, " ")), [sites(1)[2:8]; [sites(1)[1]]; sites(2)])
        end
        for method in ("exact", "kpm")
            cd(dir) do
                conf = get_dos_test_config()
                set_value!(conf, "poscar", poscar)
                set_value!(conf, "dos", "dos pdos cohp")
                set_value!(conf, "projection", "DOS", "defect")
                set_value!(conf, "primitive_poscar", "DOS", joinpath(dos_path, "POSCAR_gaas"))
                set_value!(conf, "method", "DOS", method)
                set_value!(conf, "kpm_moments", "DOS", 256)
                Hamster.main(comm, conf, rank=rank, nranks=nranks)
                h5open("dos.h5") do f
                    g = f["1"]
                    E = read(g["energies"])
                    # The antisite, its 4 As neighbours, 6 second neighbours (Ga) and the remaining 5 atoms
                    @test sort(read(g["regions"])) == [fill(-1, 5); 0; fill(1, 4); fill(2, 6)]
                    @test sort(keys(g["pdos"])) == ["bulk-As", "bulk-Ga", "defect-As", "shell1-As", "shell2-Ga"]
                    @test read(g["label_atoms/defect-As"]) == 1 && read(g["label_atoms/shell1-As"]) == 4
                    @test sum(read(g["pdos/$l"]) for l in keys(g["pdos"])) ≈ read(g["dos"]) rtol=1e-6
                    # Exact for both methods (unit vectors with kpm): 8 states per atom
                    @test integral(E, read(g["pdos/defect-As"])) ≈ 8 atol=1e-3
                    @test integral(E, read(g["pdos/shell1-As"])) ≈ 32 atol=1e-3
                    @test "defect-As-shell1-As" ∈ keys(g["cohp"])
                end
            end
        end
    end
end

@testset "Localized states" begin
    # 8 atoms (4 A, 4 B): a state spread over the A atoms (Bloch-like on one sublattice) and a state on atom 6
    types = UInt8[1, 1, 1, 1, 2, 2, 2, 2]
    Wa = zeros(8, 2); Wa[1:4, 1] .= 0.25; Wa[6, 2] = 1
    energies = collect(range(-2, 3, length=501))
    pn, ratio, wa, density = Hamster.participation(energies, [([0.0, 1.0], Wa)], types, 0.05, 1.0)
    e0 = findfirst(≈(0.0), energies); e1 = findfirst(≈(1.0), energies)
    @test pn[e0] ≈ 4 && ratio[e0] ≈ 1           # not localized, although only on the A sublattice
    @test pn[e1] ≈ 1 && ratio[e1] ≈ 0.25
    @test isnan(ratio[1]) && wa[:, e1] ≈ Wa[:, 2]
    atoms, peaks = Hamster.localized_atoms(energies, ratio, pn, wa, density, threshold=0.5)
    @test atoms == [6] && length(peaks) == 1 && peaks[1].energy ≈ 1.0
    @test isempty(Hamster.localized_atoms(energies, ratio, pn, wa, density, threshold=0.5, emax=0.5)[1])

    # Standard run of the perfect crystal: no localized states, all atoms are bulk
    mktempdir() do dir
        cd(dir) do
            conf = get_dos_test_config()
            set_value!(conf, "dos", "pdos")
            set_value!(conf, "projection", "DOS", "localized")
            Hamster.main(comm, conf, rank=rank, nranks=nranks)
            h5open("dos.h5") do f
                @test all(==(-1), read(f["1/regions"])) && keys(f["1/pdos"]) == ["bulk-As", "bulk-Ga"]
                @test isempty(read(f["1/participation/peak_energies"]))
                @test all(r -> isnan(r) || r > 0.9, read(f["1/participation/ratio"]))
            end
        end
    end
end
