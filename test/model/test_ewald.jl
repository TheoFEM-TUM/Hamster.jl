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

@testset "Ewald slab" begin
    # rock-salt slab (in-plane lattice constant 4 Å, 6 layers 2 Å apart, 40 Å vacuum) with dipoles of the upper layers
    # (anions displaced along z), placed across the periodic boundary along the slab axis
    a, nl, dz, vac = 4.0, 6, 2.0, 40.0
    c = (nl - 1) * dz + vac
    box = [a 0 0; 0 a 0; 0 0 c]
    pos = Vector{Float64}[]; q = Float64[]
    for l in 0:nl-1
        shift = l ≥ 3 ? 0.15 * (l - 2) : 0.0
        for (x, y, s) in ((0.0, 0.0, 1.0), (a / 2, a / 2, -1.0))
            qa = isodd(l) ? -s : s
            z = mod(l * dz + (qa < 0 ? shift : 0.0) + c - 5.0, c)      # the slab crosses z = 0
            push!(pos, [x + (isodd(l) ? a / 2 : 0.0), y, z]); push!(q, qa)
        end
    end
    # real-space sum in a long cell with a short in-plane period (images along a1, a2 far beyond one lattice constant)
    for rc in (12.0, 30.0)
        g = Hamster.ewald_sum(pos, q, box, rcut=rc, alpha=0.2, include_long=false, subtract_self=false).potentials
        b = zeros(length(q)); nimg = [ceil(Int, rc / norm(box[:, d])) + 1 for d in 1:3]
        for i in eachindex(pos), j in eachindex(pos), n1 in -nimg[1]:nimg[1], n2 in -nimg[2]:nimg[2], n3 in -nimg[3]:nimg[3]
            r = norm(pos[i] - pos[j] - box * [n1, n2, n3])
            1e-12 < r < rc && (b[i] += q[j] * Hamster.erfc(0.2 * r) / r)
        end
        @test g ≈ Hamster.ke .* b atol=1e-10
    end
    z, A, vacuum = Hamster.slab_heights(pos, box, 3)
    @test A ≈ a^2
    @test maximum(z) - minimum(z) < (nl - 1) * dz + 1 && vacuum > vac - 1
    @test minimum(z) ≈ 0 atol=1e-12

    # full Ewald with dipole correction = Wolf (α = 0.2, large cutoff) + plane-averaged long range, up to a constant per
    # species (the in-plane components of the smooth part, ∝ exp(-G²/4α²), are negligible for this cell). Reference:
    # real-space part of ewald_sum plus a direct reciprocal sum (more accurate than the particle mesh for this check)
    function direct_ewald(pos, q, box; α=0.35, rcut=14.0, kmax=4.0)
        φ = Hamster.ewald_sum(pos, q, box, rcut=rcut, alpha=α, include_long=false, subtract_self=true).potentials
        B = 2π * inv(box)'; V = abs(det(box))
        n = [ceil(Int, kmax / norm(B[:, d])) for d in 1:3]
        for i in -n[1]:n[1], j in -n[2]:n[2], l in -n[3]:n[3]
            (i, j, l) == (0, 0, 0) && continue
            k = B * [i, j, l]; k2 = dot(k, k); k2 > kmax^2 && continue
            ρ = sum(q[m] * cis(dot(k, pos[m])) for m in eachindex(q))
            for a in eachindex(pos)
                φ[a] += Hamster.ke * 4π / V * exp(-k2 / (4α^2)) / k2 * real(ρ * cis(-dot(k, pos[a])))
            end
        end
        return φ
    end
    ewald = direct_ewald(pos, q, box) .+ Hamster.slab_potentials(pos, q, box, 3; method="ewald", alpha=0.0)
    wolf = Hamster.ewald_sum(pos, q, box, rcut=30.0, alpha=0.2, method="wolf").potentials .+
           Hamster.slab_potentials(pos, q, box, 3; method="wolf", alpha=0.2)
    centered(φ) = (ψ = copy(φ); for s in (-1.0, 1.0); m = q .== s; ψ[m] .-= sum(φ[m]) / count(m); end; ψ)
    @test maximum(abs, centered(ewald) .- centered(wolf)) < 1e-4
    # without the slab terms, Wolf misses the potential differences between the layers that the dipoles create
    wolf0 = Hamster.ewald_sum(pos, q, box, rcut=30.0, alpha=0.2, method="wolf").potentials
    @test maximum(abs, centered(ewald) .- centered(wolf0)) > 0.1

    # potential step of a dipole layer (charges ±1, 0.5 Å apart) between points far above and below (zero charges)
    box2 = [5.0 0 0; 0 5.0 0; 0 0 100.0]
    pos2 = [[0.0, 0.0, 30.0], [2.5, 2.5, 30.5], [0.0, 0.0, 10.0], [0.0, 0.0, 50.0]]
    φ2 = Hamster.slab_potentials(pos2, [1.0, -1.0, 0.0, 0.0], box2, 3; method="wolf", alpha=0.5)
    @test φ2[4] - φ2[3] ≈ 4π * Hamster.ke * (-1.0 * 0.5) / 25.0 rtol=1e-10
    @test_throws ErrorException Hamster.slab_potentials(pos2, ones(4), box2, 4; method="wolf", alpha=0.5)

    # charged cells: Wolf allowed, full Ewald not
    @test length(Hamster.ewald_sum(pos2, [1.0, 0.0, 0.0, 0.0], box2, rcut=10.0, alpha=0.3, method="wolf").potentials) == 4
    @test_throws ErrorException Hamster.ewald_sum(pos2, [1.0, 0.0, 0.0, 0.0], box2, rcut=10.0)
end

@testset "Ewald charge compensation" begin
    # layers along z: one atom per layer (the plane-averaged term only sees the heights)
    function stack(σs, spacings; a=4.0, vac=40.0)
        z = cumsum([0.0; spacings]); c = z[end] + vac
        pos = [[0.0, 0.0, zi + 5.0] for zi in z]
        return pos, collect(Float64, σs), [a 0 0; 0 a 0; 0 0 c], z
    end
    plane(pos, q, box) = Hamster.slab_potentials(pos, q, box, 3; method="wolf", alpha=0.5)

    # symmetric slab with layers ±σ, the same termination on both sides (net charge -σ): half of σ on each surface layer
    σ = 0.6; nl = 17
    pos, q, box, z = stack([isodd(l) ? -σ : σ for l in 1:nl], fill(3.0, nl - 1))
    @test Hamster.slab_layers(z) == 1:nl
    c = Hamster.surface_compensation(z, q)
    @test c[1] ≈ σ / 2 && c[end] ≈ σ / 2 && all(iszero, c[2:end-1])
    @test sum(q .+ c) ≈ 0 atol=1e-12
    # no drift of the potential through the interior (layers of the same kind two periods apart), unlike the uniform
    # background, which leaves the surfaces charged
    φc = plane(pos, q .+ c, box); φg = plane(pos, q, box)
    @test abs(φc[5] - φc[7]) < 1e-6 && abs(φc[6] - φc[10]) < 1e-6
    @test abs(φg[5] - φg[7]) > 0.1

    # polar stacking with alternating spacings a, b and no net charge: surface charge σ a / (a + b) of the polarization
    a, b = 1.5, 3.0; nl = 16
    pos, q, box, z = stack([isodd(l) ? -σ : σ for l in 1:nl], [isodd(k) ? a : b for k in 1:nl-1])
    c = Hamster.surface_compensation(z, q)
    @test c[1] ≈ σ * a / (a + b) && c[end] ≈ -σ * a / (a + b)
    φc = plane(pos, q .+ c, box)
    @test abs(φc[5] - φc[7]) < 1e-6 && abs(φc[6] - φc[10]) < 1e-6
    φu = plane(pos, q, box)
    @test abs(φu[5] - φu[7]) > 0.1
    # neutral layers (formal charges): nothing to compensate
    @test all(iszero, Hamster.surface_compensation(z, zeros(nl)))
    @test_throws ErrorException Hamster.surface_compensation([0.0, 3.0], [1.0, -1.0])

    # several atoms per layer, slightly buckled: layers by the gaps along the axis
    zb = [0.0, 0.3, 3.0, 3.2, 6.1, 6.0]
    @test Hamster.slab_layers(zb) == [1, 1, 2, 2, 3, 3]
    @test Hamster.surface_compensation(zb, [-0.5, -0.5, 1.0, 1.0, -0.5, -0.5]) ≈ zeros(6) atol=1e-12    # neutral, symmetric
    cb = Hamster.surface_compensation(zb, [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5])                     # net charge -1
    @test cb ≈ [0.25, 0.25, 0.0, 0.0, 0.25, 0.25]

    # local: rock salt (4 × 4 × 4 sites) with one anion missing; the six cations around it take the compensation
    d = 2.8; n = 4
    sites = [[i, j, k] for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]
    vacancy = findfirst(s -> s == [1, 2, 2], sites)
    keep = setdiff(eachindex(sites), [vacancy])
    pos = [d .* Float64.(sites[i]) .+ 0.05 .* [sin(i), cos(2i), sin(3i)] for i in keep]     # distorted
    qs = [iseven(sum(sites[i])) ? 0.8 : -0.8 for i in keep]
    types = [q > 0 ? UInt8(11) : UInt8(17) for q in qs]
    lat = d * n * Matrix(1.0I, 3, 3)
    cn = Hamster.coordination_numbers(pos, types, qs, lat)
    neighbors_of_vacancy = [k for (k, i) in enumerate(keep) if sum(abs, mod.(sites[i] .- sites[vacancy] .+ n ÷ 2, n) .- n ÷ 2) == 1]
    @test length(neighbors_of_vacancy) == 6 && all(==(5), cn[neighbors_of_vacancy])
    @test count(==(6), cn) == length(keep) - 6
    cl = Hamster.local_compensation(pos, types, qs, lat)
    @test cl[neighbors_of_vacancy] ≈ fill(-sum(qs) / 6, 6)
    @test sum(qs .+ cl) ≈ 0 atol=1e-12
    @test count(!iszero, cl) == 6
    # neutral cell: nothing to compensate; a charged cell without missing neighbors cannot be compensated locally
    full = [d .* Float64.(s) for s in sites]; qf = [iseven(sum(s)) ? 0.8 : -0.8 for s in sites]; tf = [q > 0 ? UInt8(11) : UInt8(17) for q in qf]
    @test all(iszero, Hamster.local_compensation(full, tf, qf, lat))
    @test_throws ErrorException Hamster.local_compensation(full, tf, qf .+ 0.01, lat)

    @test Hamster.compensation_charges("global", pos, types, qs, lat) == zeros(length(qs))
    @test_throws ErrorException Hamster.compensation_charges("surface", pos, types, qs, lat)
    @test_throws ErrorException Hamster.compensation_charges("nearby", pos, types, qs, lat)
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
    @test Hamster.mean_uncertainty(30.0, 9, 80.0, 72) ≈ 10
    @test Hamster.mean_uncertainty(NaN, 1, 80.0, 16) ≈ 20
    @test isnan(Hamster.mean_uncertainty(NaN, 1, 80.0, 1))

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
            @test occursin(r"σ_strc\s+δφ̄", out)

            # slab_axis: the plane-averaged long range changes the potentials and is reported
            wolf0 = EwaldOnsites([strc], [basis], comm, conf, method="wolf", rcut=10.0, alpha=0.3, write_summary=false)
            wolf3 = EwaldOnsites([strc], [basis], comm, conf, method="wolf", rcut=10.0, alpha=0.3, slab_axis=3, label="slab")
            @test maximum(abs, wolf3.potentials[1] .- wolf0.potentials[1]) > 1e-4
            @test occursin("slab (vacuum along a3):", read("hamster.out", String))
            # slab_scale multiplies the slab term
            wolf3h = EwaldOnsites([strc], [basis], comm, conf, method="wolf", rcut=10.0, alpha=0.3, slab_axis=3, slab_scale=0.5,
                                  write_summary=false)
            @test wolf3h.potentials[1] .- wolf0.potentials[1] ≈ 0.5 .* (wolf3.potentials[1] .- wolf0.potentials[1]) rtol=1e-10 atol=1e-12
            # charge compensation: default surface with slab_axis (neutral layers here: no change), reported; local in a
            # neutral cell changes nothing
            @test occursin("net charge compensated:", read("hamster.out", String))
            wolfg = EwaldOnsites([strc], [basis], comm, conf, method="wolf", rcut=10.0, alpha=0.3, slab_axis=3,
                                 charge_compensation="global", write_summary=false)
            @test wolfg.potentials[1] ≈ wolf3.potentials[1] atol=1e-10
            wolfl = EwaldOnsites([strc], [basis], comm, conf, method="wolf", rcut=10.0, alpha=0.3, charge_compensation="local",
                                 write_summary=false)
            @test wolfl.potentials[1] ≈ wolf0.potentials[1] atol=1e-10
            @test_throws ErrorException EwaldOnsites([strc], [basis], comm, conf, method="wolf", rcut=10.0, alpha=0.3,
                                                     charge_compensation="surface", write_summary=false)
            set_value!(conf, "charge_compensation", "Ewald", "Local")
            @test Hamster.get_ewald_charge_compensation(conf) == "local"
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

    # Zahn (damped shifted force): potential and its derivative vanish at the cutoff
    α, rc = 0.3, 9.0
    @test abs(Hamster.zahn_kernel(rc, α, rc)) < 1e-14
    h = 1e-6
    @test abs((Hamster.zahn_kernel(rc + h, α, rc) - Hamster.zahn_kernel(rc - h, α, rc)) / 2h) < 1e-8
    @test abs(Hamster.wolf_kernel(rc, α, rc)) < 1e-14

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

    # Wolf without rcut and alpha: from the size of the training cells in a fit, error otherwise
    @test Hamster.cell_size([2.0 0 0; 0 3.0 0; 0 0 4.0]) ≈ 4.0
    @test Hamster.cell_size([1.0 0.5 0; 0 1.0 0; 0 0 1.0]) ≈ 1.0     # widths 1/norm([1, -0.5]), 1 and 1
    default = cspbbr3_conf()
    set_value!(default, "method", "Ewald", "wolf")
    @test_throws ErrorException Hamster.apply_ewald_settings!(default, strcs=[strc])
    Hamster.apply_ewald_settings!(default, strcs=[strc], comm=comm, training=true)
    L = Hamster.cell_size(strc.lattice)
    @test Hamster.get_ewald_rcut(default) ≈ 1.5 * L
    @test Hamster.ewald_alpha("wolf", Hamster.get_ewald_rcut(default), Hamster.get_ewald_alpha(default)) ≈ 2 / L
    explicit = cspbbr3_conf()
    set_value!(explicit, "method", "Ewald", "wolf")
    set_value!(explicit, "rcut", "Ewald", 12.0)
    Hamster.apply_ewald_settings!(explicit, strcs=[strc], training=true)
    @test Hamster.get_ewald_rcut(explicit) == 12.0 && Hamster.ewald_alpha("wolf", 12.0, Hamster.get_ewald_alpha(explicit)) ≈ 0.25
    explicit_alpha = cspbbr3_conf()
    set_value!(explicit_alpha, "method", "Ewald", "wolf")
    set_value!(explicit_alpha, "alpha", "Ewald", 0.25)
    Hamster.apply_ewald_settings!(explicit_alpha, strcs=[strc], training=true)
    @test Hamster.get_ewald_alpha(explicit_alpha) == 0.25 && Hamster.get_ewald_rcut(explicit_alpha) ≈ 12.0
    only_rcut = cspbbr3_conf()
    set_value!(only_rcut, "method", "Ewald", "wolf")
    set_value!(only_rcut, "rcut", "Ewald", 15.0)
    Hamster.apply_ewald_settings!(only_rcut, strcs=[strc])                    # no fit: alpha = 3 / rcut when the model is built
    @test Hamster.ewald_alpha("wolf", Hamster.get_ewald_rcut(only_rcut), Hamster.get_ewald_alpha(only_rcut)) ≈ 0.2
end
