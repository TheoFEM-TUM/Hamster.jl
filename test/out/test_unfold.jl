@testset "Band unfolding" begin
    path = joinpath(@__DIR__, "..", "model", "test_files")
    function gaas_config(poscar; sparse=false)
        conf = get_empty_config()
        set_value!(conf, "poscar", poscar)
        for (element, alpha) in ("Ga" => 9, "As" => 13)
            set_value!(conf, "orbitals", element, "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
            set_value!(conf, "n", element, 3)
            set_value!(conf, "alpha", element, alpha)
        end
        set_value!(conf, "NNaxes", "As", true)
        set_value!(conf, "load_rllm", true)
        set_value!(conf, "rllm_file", joinpath(path, "rllm_true.dat"))
        set_value!(conf, "rcut", 5)
        set_value!(conf, "init_params", joinpath(path, "params.dat"))
        set_value!(conf, "verbosity", 0)
        if sparse
            set_value!(conf, "sp_mode", true); set_value!(conf, "sp_diag", true)
            set_value!(conf, "neig", 12); set_value!(conf, "eig_target", 2.0)
        end
        return conf
    end
    model(conf) = (strc = Structure(conf); basis = Basis(strc, conf);
                   (strc, basis, EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)))

    mktempdir() do dir
        # Ideal 2x2x2 supercell of the primitive GaAs cell
        pc = Hamster.read_poscar(joinpath(path, "POSCAR_gaas"))
        sc_poscar = joinpath(dir, "POSCAR_sc")
        open(sc_poscar, "w") do f
            println(f, "GaAs 2x2x2\n1.0")
            for a in eachcol(pc.lattice); println(f, join(2 .* a, " ")); end
            println(f, "Ga As\n8 8\nDirect")
            for atom in 1:2, n1 in 0:1, n2 in 0:1, n3 in 0:1
                println(f, join((pc.rs_atom[:, atom] .+ [n1, n2, n3]) ./ 2, " "))
            end
        end
        ks = [0 0.25 0.5 0.5 0.3 0.125; 0 0 0 0.5 0.1 0.875; 0 0 0 0 0.2 0.5]

        conf_pc = gaas_config(joinpath(path, "POSCAR_gaas"))
        _, _, ham_pc = model(conf_pc)
        Es_pc, _ = diagonalize(get_hamiltonian(ham_pc, 1, ks))

        conf_sc = gaas_config(sc_poscar)
        strc, basis, ham_sc = model(conf_sc)
        umap = Hamster.UnfoldMap(strc, basis, pc.lattice, pc.rs_atom, pc.atom_types)
        @test umap.Nc == 8 && umap.Nj == 8 && umap.M == 2I
        Ks, kgroups = Hamster.supercell_kpoints(ks, umap)
        @test size(Ks, 2) == 4    # Γ, (0.5,0,0), (0.5,0.5,0) fold onto Γ
        @test sort(kgroups[1]) == [1, 3, 4]

        # Perfect supercell: weights 0 or 1, eight states with weight 1 at the eigenvalues of the primitive cell
        Es, Ws = Hamster.unfold_structure(ham_sc, 1, umap, ks, conf_sc)
        @test size(Es) == (64, 6)
        # (degenerate states can mix, so the weights are summed over groups of degenerate states)
        degenerate_groups(E) = (groups = [[1]]; for n in 2:length(E); E[n] - E[n-1] < 1e-6 ? push!(groups[end], n) : push!(groups, [n]); end; groups)
        @test all(k -> sum(Ws[:, k]) ≈ 8, axes(Ws, 2))
        for k in axes(ks, 2)
            groups = degenerate_groups(Es[:, k])
            Wg = [sum(Ws[g, k]) for g in groups]
            @test all(w -> isapprox(w, round(w), atol=1e-8), Wg)
            unfolded = reduce(vcat, [fill(Es[first(g), k], round(Int, w)) for (g, w) in zip(groups, Wg)])
            @test unfolded ≈ Es_pc[:, k] atol=1e-6
        end

        # Same weights as the pair sum over all copies of an orbital (earlier implementation)
        vs = randn(ComplexF64, 64, 3); k = ks[:, 5]
        W = zeros(3); Hamster.spectral_weights!(W, vs, k, umap)
        W_pairs = [real(sum(conj(vs[i, N]) * vs[i2, N] * cis(2π * dot(k, umap.T[:, i2] - umap.T[:, i]))
                            for i in 1:64, i2 in 1:64 if umap.group[i] == umap.group[i2])) / umap.Nc for N in 1:3]
        @test W ≈ W_pairs

        # Partial diagonalization (shift-invert): the 12 states closest to the target
        conf_sp = gaas_config(sc_poscar, sparse=true)
        _, _, ham_sp = model(conf_sp)
        Es_sp, Ws_sp = Hamster.unfold_structure(ham_sp, 1, umap, ks, conf_sp)
        @test size(Es_sp) == (12, 6)
        for k in axes(ks, 2)
            @test all(N -> minimum(abs.(Es[:, k] .- Es_sp[N, k])) < 1e-6, axes(Es_sp, 1))
            # Weight of every complete group of degenerate states = weight of the full diagonalization
            for g in degenerate_groups(Es[:, k])
                inside = findall(N -> abs(Es_sp[N, k] - Es[first(g), k]) < 1e-6, axes(Es_sp, 1))
                length(inside) == length(g) && @test sum(Ws_sp[inside, k]) ≈ sum(Ws[g, k]) atol=1e-6
            end
        end

        # Shift-invert solvers: same eigenvalues; KrylovKit gives orthonormal eigenvectors also for degenerate states
        Hk = get_hamiltonian(ham_sp, 1, ks[:, 1:1])[1]
        E_ar, _ = diagonalize(Hk, Neig=12, target=2.0, method="shift-invert-arpack")
        E_kr, V_kr = diagonalize(Hk, Neig=12, target=2.0, method="shift-invert")
        @test sort(E_ar) ≈ E_kr atol=1e-8
        @test norm(V_kr' * V_kr - I) < 1e-8
        @test norm(Hk * V_kr - V_kr * Diagonal(E_kr)) < 1e-8
        @test_throws ErrorException diagonalize(Hk, Neig=4, target=2.0, method="lapack")
        @test Hamster.unfold_diag_method(conf_sp) == "shift-invert"
        set_value!(conf_sp, "diag_method", "shift-invert-arpack")
        @test Hamster.unfold_diag_method(conf_sp) == "shift-invert"
        set_value!(conf_sp, "diag_method", "shift-invert")

        # Spectral function: integral over the energy = sum of the weights
        energies = range(minimum(Es) - 1, maximum(Es) + 1, length=4000)
        A = Hamster.spectral_function(Es, Ws, energies, 0.05)
        @test all(k -> isapprox(sum(A[:, k]) * step(energies), 8, rtol=1e-3), axes(A, 2))

        # KPM: moments of the Bloch starting vectors = moments of the exact eigenvalues and weights
        M = 40
        μs, centers, halfwidths = Hamster.kpm_unfold_structure(ham_sc, 1, umap, ks, M)
        for k in axes(ks, 2)
            x = (Es[:, k] .- centers[k]) ./ halfwidths[k]
            @test maximum(abs, x) < 1
            μ_exact = [sum(Ws[:, k] .* cos.(m .* acos.(x))) for m in 0:M-1]
            @test μs[:, k] ≈ μ_exact atol=1e-8
        end
        A_kpm = Hamster.kpm_spectral_function(Hamster.kpm_unfold_structure(ham_sc, 1, umap, ks, 512)..., energies)
        @test all(k -> isapprox(sum(A_kpm[:, k]) * step(energies), 8, rtol=1e-2), axes(A_kpm, 2))

        # Energy window of the dense diagonalization: the same states and weights as the full diagonalization
        Hk_sc = get_hamiltonian(ham_sc, 1, ks[:, 5:6])
        E_w, v_w = diagonalize(Hk_sc, vectors=true, window=(-1.0, 3.0))
        E_full, _ = diagonalize(Hk_sc)
        for k in 1:2
            @test filter(isfinite, E_w[:, k]) ≈ filter(E -> -1 ≤ E ≤ 3, E_full[:, k]) atol=1e-10
        end
        @test size(v_w, 2) == maximum(k -> count(isfinite, E_w[:, k]), 1:2) < 64
        conf_w = gaas_config(sc_poscar); set_value!(conf_w, "diag_emin", -1.0); set_value!(conf_w, "diag_emax", 3.0)
        Es_w, Ws_w = Hamster.unfold_structure(ham_sc, 1, umap, ks, conf_w)
        for k in axes(ks, 2), g in degenerate_groups(Es[:, k])
            -1 < Es[first(g), k] - 1e-6 && Es[last(g), k] + 1e-6 < 3 || continue
            inside = findall(E -> isfinite(E) && abs(E - Es[first(g), k]) < 1e-6, Es_w[:, k])
            @test length(inside) == length(g)
            @test sum(Ws_w[inside, k]) ≈ sum(Ws[g, k]) atol=1e-8
        end
        @test any(isnan, Es_w) && all(Ws_w[isnan.(Es_w)] .== 0)
        @test_throws ErrorException Hamster.diag_window((set_value!(conf_w, "diag_emax", -2.0); conf_w))

        # Device code path (ArrayDevice with host arrays): same weights and moments as the host path
        ham_dev = EffectiveHamiltonian([strc], [basis], comm, conf_sc, rank=rank, nranks=nranks, device=Hamster.ArrayDevice(Array))
        Es_dev, Ws_dev = Hamster.unfold_structure(ham_dev, 1, umap, ks, conf_sc)
        @test Es_dev ≈ Es atol=1e-8
        @test all(k -> sum(Ws_dev[:, k]) ≈ 8, axes(Ws_dev, 2))
        @test Hamster.kpm_unfold_structure(ham_dev, 1, umap, ks, M)[1] ≈ μs atol=1e-8

        # Effective bands of the ideal supercell: the bands of the primitive cell
        energies_fine = range(minimum(Es) - 1, maximum(Es) + 1, length=100000)
        Eb, lower, upper = Hamster.effective_bands(energies_fine, Hamster.spectral_function(Es, Ws, energies_fine, 0.001))
        @test size(Eb) == (8, 6)
        @test maximum(abs, Eb .- Es_pc) < 0.002     # bands closer than σ shift each other by up to ~σ
        @test all(lower .< Eb .< upper) && maximum(upper .- lower) < 0.01

        # Non-integer weight below the gap (1.8 of 2 valence bands at this k): band edges and ribbons stay out of the gap
        energies_gap = range(-3, 3, length=6001)
        A_gap = [1.8 * exp(-(e + 1)^2 / 0.02) + 2.2 * exp(-(e - 1)^2 / 0.02) for e in energies_gap, _ in 1:1] ./ sqrt(0.02π)
        Eb, lower, upper = Hamster.effective_bands(energies_gap, A_gap, nbands=4)
        @test all(<(-0.5), [Eb[1:2]; lower[1:2]; upper[1:2]]) && all(>(0.5), [Eb[3:4]; lower[3:4]; upper[3:4]])
        @test Hamster.gap_ranges(energies_gap, vec(A_gap), 0.01, 0.1) |> length == 3

        # Band-resolved weights: projections onto the band states of the primitive cell (same model)
        Epc, Xpc = Hamster.primitive_bands(conf_sc, joinpath(path, "POSCAR_gaas"), ks)
        @test Epc ≈ Es_pc atol=1e-10
        Es_b, Ws_b, Wb = Hamster.unfold_structure(ham_sc, 1, umap, ks, conf_sc, bands=Xpc, band_energies=Epc)
        @test dropdims(sum(Wb, dims=1), dims=1) ≈ Ws_b atol=1e-10
        for k in axes(ks, 2), n in 1:8
            @test sum(Wb[n, :, k]) ≈ 1 atol=1e-10                         # ideal: every band is one supercell state
            @test sum(Wb[n, :, k] .* Es_b[:, k]) ≈ Epc[n, k] atol=1e-6    # at the energy of the band (averaged within degenerate bands)
        end
        μs_b, c_b, h_b, μb = Hamster.kpm_unfold_structure(ham_sc, 1, umap, ks, M, bands=Xpc, band_energies=Epc)
        @test μs_b ≈ μs atol=1e-8                                         # the same space of starting vectors
        @test all(μb[1, :, :] .≈ 1)
        @test c_b' .+ h_b' .* μb[2, :, :] ≈ Epc atol=1e-6                # mean energy of every band (first moment)
        med, lo, hi, mean, sd, weight = Hamster.band_resolved_bands(energies_fine,
            permutedims(reduce((a, b) -> cat(a, b, dims=3), [Hamster.spectral_function(Es_b, Wb[n, :, :], energies_fine, 0.001) for n in 1:8]), (1, 3, 2)))
        @test maximum(abs, med .- Epc) < 1e-4 && maximum(abs, mean .- Epc) < 1e-4 && all(weight .≈ 1)
        @test all(lo .< med .< hi) && maximum(sd) < 0.0011

        # Standard run with an Unfold block (exact and kpm)
        h5open(joinpath(dir, "kpoints_pc.h5"), "w") do f; f["kpoints"] = ks; end
        for method in ("exact", "kpm")
            cd(dir) do
                conf = gaas_config(sc_poscar)
                set_value!(conf, "poscar", "Unfold", joinpath(path, "POSCAR_gaas"))
                set_value!(conf, "kpoints", "Unfold", joinpath(dir, "kpoints_pc.h5"))
                set_value!(conf, "method", "Unfold", method)
                set_value!(conf, "kpm_moments", "Unfold", 256)
                set_value!(conf, "band_resolved", "Unfold", true)
                Hamster.main(comm, conf, rank=rank)
                h5open("unfold.h5") do f
                    @test read(f["kpoints"]) == ks
                    @test size(read(f["spectral_function"])) == (1000, 6)
                    @test read(attributes(f)["method"]) == method
                    @test haskey(f, "effective_bands") && size(read(f["effective_bands/energies"])) == (8, 6)
                    @test haskey(f, "1/" * (method == "exact" ? "Ws" : "moments"))
                    @test size(read(f["band_resolved/spectral_function"])) == (1000, 8, 6)
                    @test read(f["band_resolved/primitive_bands"]) ≈ Es_pc atol=1e-10
                    @test all(isapprox.(read(f["band_resolved/weight"]), 1, atol=0.01))
                    # exact: on the energy grid; kpm: from the moments (exact): the ideal bands have no width
                    @test maximum(abs, read(f["band_resolved/mean"]) .- Es_pc) < (method == "exact" ? 0.01 : 1e-6)
                    method == "kpm" && @test maximum(read(f["band_resolved/std"])) < 1e-3
                end
                method == "exact" && @test maximum(abs, h5read("unfold.h5", "effective_bands/energies") .- Es_pc) < 2 * 0.05   # ~σ
            end
        end

        # Defects: vacancy (As missing), antisite (As on a Ga site), interstitial (Ga on the empty tetrahedral site)
        function defect_poscar(name, ga, as)
            file = joinpath(dir, name)
            open(file, "w") do f
                println(f, "GaAs defect\n1.0"); for a in eachcol(pc.lattice); println(f, join(2 .* a, " ")); end
                println(f, "Ga As\n$(length(ga)) $(length(as))\nDirect")
                foreach(r -> println(f, join(r, " ")), vcat(ga, as))
            end
            return file
        end
        sites(atom) = [(pc.rs_atom[:, atom] .+ [n1, n2, n3]) ./ 2 for n1 in 0:1 for n2 in 0:1 for n3 in 0:1]
        defects = [("vacancy", sites(1), sites(2)[1:7], (0, 0, 1), 7.5),
                   ("antisite", sites(1)[2:8], [[sites(1)[1]]; sites(2)], (1, 0, 0), 8.0),
                   ("interstitial", [sites(1); [fill(0.375, 3)]], sites(2), (0, 1, 0), 8.0)]
        for (name, ga, as, (nsub, nunmapped, nvacant), total) in defects
            conf_d = gaas_config(defect_poscar("POSCAR_$name", ga, as))
            strc_d, basis_d, ham_d = model(conf_d)
            umap_d = Hamster.UnfoldMap(strc_d, basis_d, pc.lattice, pc.rs_atom, pc.atom_types)
            @test length.((umap_d.substituted, umap_d.unmapped, umap_d.vacant)) == (nsub, nunmapped, nvacant)
            Es_d, Ws_d = Hamster.unfold_structure(ham_d, 1, umap_d, ks, conf_d)
            @test all(k -> sum(Ws_d[:, k]) ≈ total, axes(Ws_d, 2))
            @test all(w -> -1e-10 < w < 1 + 1e-10, Ws_d)
            μ_d = Hamster.kpm_unfold_structure(ham_d, 1, umap_d, ks, 4)[1]
            @test all(k -> μ_d[1, k] ≈ total, axes(μ_d, 2))
        end

        # Vacancy position; split interstitial (a second As 0.3 Å from an As site): the closer atom is mapped
        strc_v, basis_v, _ = model(gaas_config(defect_poscar("POSCAR_vac", sites(1), sites(2)[1:7])))
        umap_v = Hamster.UnfoldMap(strc_v, basis_v, pc.lattice, pc.rs_atom, pc.atom_types)
        @test umap_v.vacant == [2] && umap_v.vacant_positions[:, 1] ≈ 2 .* (pc.lattice * sites(2)[8])
        shift = pc.lattice \ [0.3, 0, 0] ./ 2
        strc_s, basis_s, _ = model(gaas_config(defect_poscar("POSCAR_split", sites(1), [sites(2); [sites(2)[3] .+ shift]])))
        umap_s = Hamster.UnfoldMap(strc_s, basis_s, pc.lattice, pc.rs_atom, pc.atom_types)
        @test umap_s.unmapped == [17] && isempty(umap_s.vacant) && umap_s.site[11] == 2
        levels, owner = Hamster.defect_regions(strc_s, umap_s, pc.lattice, pc.rs_atom, shells=1)
        @test levels[17] == 0 && levels[11] == 1 && count(==(1), levels) == 5    # the As of the pair and 4 Ga
        @test all(==(1), owner[levels .≥ 0])
        # Two As vacancies (not neighbours): two defects, every Ga next to a vacancy belongs to the closer one
        strc_2, basis_2, _ = model(gaas_config(defect_poscar("POSCAR_2vac", sites(1), sites(2)[[1, 2, 3, 5, 6, 7]])))
        umap_2 = Hamster.UnfoldMap(strc_2, basis_2, pc.lattice, pc.rs_atom, pc.atom_types)
        levels_2, owner_2 = Hamster.defect_regions(strc_2, umap_2, pc.lattice, pc.rs_atom, shells=1)
        @test length(umap_2.vacant) == 2 && sort(unique(owner_2[levels_2 .== 1])) == [1, 2]
        @test all(==(0), owner_2[levels_2 .== -1]) && all(>(0), owner_2[levels_2 .== 1])
        @test all(==(1), Hamster.defect_regions(strc_2, umap_2, pc.lattice, pc.rs_atom, shells=1, separate=false)[2][levels_2 .== 1])
        @test Hamster.region_name(1, "defect", 2, 2) == "defect2-shell1" && Hamster.region_name(0, "defect", 1, 1) == "defect"
        # Displaced positions: an atom moved by 0.4 Å is still mapped with tol = 1, not with the default tol
        strc_m = Hamster.Structure(strc.system, strc.lattice, strc.Rs,
                                   [Hamster.Ion(ion.type, ion.pos, i == 1 ? [0.4, 0, 0] : zeros(3)) for (i, ion) in enumerate(strc.ions)],
                                   strc.point_grid)
        @test isempty(Hamster.UnfoldMap(strc_m, basis, pc.lattice, pc.rs_atom, pc.atom_types, displaced=true, tol=1.0).unmapped)
        @test Hamster.UnfoldMap(strc_m, basis, pc.lattice, pc.rs_atom, pc.atom_types, displaced=true, tol=0.3).unmapped == [1]

        # Mapping errors
        @test_throws ErrorException Hamster.UnfoldMap(strc, basis, 1.5 .* pc.lattice, pc.rs_atom, pc.atom_types)
        @test_throws ErrorException Hamster.UnfoldMap(strc, basis, pc.lattice, pc.rs_atom .+ 0.1, pc.atom_types, tol=0.1)
    end
end
