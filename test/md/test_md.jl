using Random, JLArrays

const MDH = Hamster

"""
Config of the MD tests: CsPbBr3 (TB parameters of the calc tests) with Wolf onsite corrections, valence electrons and a
reference structure of `supercell` primitive cells.
"""
function md_test_config(; supercell=[1, 1, 1], soc=false, kgrid=[1, 1, 1])
    path = joinpath(@__DIR__, "..", "calc", "test_files")
    conf = get_empty_config()
    set_value!(conf, "rcut", 6.2)
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_cspbbr3"))
    set_value!(conf, "init_params", joinpath(path, "params_cspbbr3.dat"))
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "soc", soc)
    for (el, α, orbs, q, nv) in (("Cs", 3.77385, "s", 1.0, 0), ("Pb", 29.83655, "s px py pz", 1.2, 2), ("Br", 16.93986, "px py pz", -0.7333333333333334, 6))
        set_value!(conf, "alpha", el, α); set_value!(conf, "orbitals", el, orbs); set_value!(conf, "qeff", el, q)
        set_value!(conf, "valence", el, nv); set_value!(conf, "charge_scale", el, 0.3)
    end
    set_value!(conf, "method", "Ewald", "wolf"); set_value!(conf, "rcut", "Ewald", 9.0); set_value!(conf, "alpha", "Ewald", 0.3)
    set_value!(conf, "kt", "MD", 0.2)
    set_value!(conf, "kgrid", "MD", join(kgrid, " "))
    set_value!(conf, "supercell", "MD", join(supercell, " "))
    return conf
end

"""
A nuclear term with smooth random coefficients for the species pairs of CsPbBr3.
"""
function md_test_repulsive(conf; seed=2)
    types = UInt8[82, 55, 35]
    rmin = Dict((a, b) => 2.2 for a in types for b in types if a ≤ b)
    rep = MDRepulsive(types, rmin, conf, coulomb=true)
    rng = Xoshiro(seed)
    rep.coeffs .= 0.05 .* randn(rng, length(rep.coeffs))
    rep.coeffs[1:3] .= [-3.0, -1.0, -2.0]
    rep.coeffs[end] = 0.3
    return rep
end

displaced(x, σ; seed=1) = x .+ σ .* randn(Xoshiro(seed), size(x))

function fd_force_check(pot, x; h=1e-4, seed=3)
    res = md_compute!(pot, x)
    u = randn(Xoshiro(seed), size(x)); u ./= norm(u)
    fd = -(md_compute!(pot, x .+ h .* u).E - md_compute!(pot, x .- h .* u).E) / 2h
    return dot(res.forces, u), fd
end

@testset "MD utilities" begin
    a, b, μ, kT = 5.0, 1.0, 0.7, 0.2
    c = MDH.chebyshev_coefficients(ε -> MDH.grand_potential(ε, μ, kT), 300, a, b)
    tab = MDH.ChebyshevTable(300, a, b)
    @test MDH.grand_potential_coefficients(tab, μ, kT) ≈ c
    d = MDH.chebyshev_derivative(c, a)
    for ε in (-3.5, 0.0, 0.7, 1.1, 5.9)
        x = (ε - b) / a
        @test MDH.chebyshev_series(c, x) ≈ MDH.grand_potential(ε, μ, kT) atol=1e-10
        @test MDH.chebyshev_series(d, x) ≈ MDH.fermi(ε, μ, kT) atol=1e-8
    end
    @test MDH.bisect(x -> x^3 - 2, 0.0, 1.0) ≈ cbrt(2) atol=1e-10

    # Neighbor list against a brute-force search in a small skewed cell (several images of the same atom)
    L = [4.0 1.0 0.5; 0.0 3.5 0.8; 0.0 0.0 5.0]
    x = L * rand(Xoshiro(1), 3, 7)
    rc = 6.0
    nl = MDH.NeighborList(x, L, rc)
    brute = Set{Tuple{Int, Int, NTuple{3, Int}}}()
    for i in 1:7, j in 1:7, n1 in -4:4, n2 in -4:4, n3 in -4:4
        (i == j && n1 == n2 == n3 == 0) && continue
        norm(x[:, j] + L * [n1, n2, n3] - x[:, i]) < rc && push!(brute, (i, j, (n1, n2, n3)))
    end
    @test Set((Int(nl.i[p]), Int(nl.j[p]), Tuple(Int.(nl.n[p]))) for p in eachindex(nl.i)) == brute
    rev = MDH.reverse_pairs(nl)
    @test all(nl.i[rev[p]] == nl.j[p] && nl.n[rev[p]] == -nl.n[p] for p in eachindex(nl.i))

    # B-splines: partition of unity in the spline range
    rep = MDRepulsive(UInt8[1], Dict((0x01, 0x01) => 1.0), get_empty_config(), coulomb=false)
    rep.coeffs .= 0; rep.coeffs[2:end] .= 1
    @test MDH.pair_energy(rep, 1, 3.0)[1] ≈ 1.0
    @test MDH.pair_energy(rep, 1, rep.rcut + 0.1)[1] == 0.0
end

@testset "MD Hamiltonian" begin
    for soc in (false, true)
        conf = md_test_config(supercell=[2, 1, 1], soc=soc)
        strc = md_reference_structure(conf)
        pot = MDPotential(strc, conf, comm, electrons="exact")
        x = displaced(pot.tb.reference, 0.15)
        Nat = length(pot.tb.types)
        # Hamiltonian of the geometry tensor for the same displacements
        strc_x = MDH.Structure(strc.Rs, pot.tb.reference, pot.tb.reference .- x, MDH.get_ion_types(strc.ions), Matrix(pot.tb.lattice), conf)
        basis = MDH.Basis(strc_x, conf)
        model = TBModel(strc_x, basis, conf)
        Href = real(sum(MDH.get_hr(model.hs, model.params, MDH.Dense())))
        MDH.tb_blocks!(pot.hblk, pot.tb, x)
        T = soc ? ComplexF64 : Float64
        Hmd = MDH.dense_hamiltonian!(zeros(T, pot.pat.N, pot.pat.N), pot.pat, pot.tb, pot.hblk, zeros(Nat), SVector(0.0, 0.0, 0.0))
        if soc
            Hsoc = MDH.dense_hamiltonian!(zeros(T, pot.pat.N, pot.pat.N), pot.pat, pot.tb, zero(pot.hblk), zeros(Nat), SVector(0.0, 0.0, 0.0))
            Hon = kron(cat([pot.tb.onsite[t] for t in pot.tb.types]...; dims=(1, 2)), [1 0; 0 1])
            Href = MDH.apply_spin_basis(Href)
            Hmd = Hmd - Hsoc + Hon
        end
        # equal up to the rounding of the bond-frame angle in the geometry tensor (cos θ to five digits)
        @test maximum(abs, Hmd - Href) < 1e-4 * maximum(abs, Href)

        # Wolf onsite shifts against EwaldOnsites of a single structure
        if !soc
            ew = EwaldOnsites([strc_x], [basis], comm, conf, write_summary=false)
            nl = MDH.NeighborList(x, pot.tb.lattice, pot.wolf.rc; rmin=1e-8)
            Δε = MDH.wolf_onsites(pot.wolf, MDH.wolf_potentials!(zeros(Nat), pot.wolf, nl, x, pot.tb.lattice))
            Δε_ref = [-ew.params[findfirst(==(ew.types_per_strc[1][i]), ew.param_labels)] * ew.potentials[1][i] for i in 1:Nat]
            @test Δε ≈ Δε_ref atol=1e-10
        end
    end
end

@testset "MD forces" begin
    # Exact solver with k-points, Wolf onsites and nuclear term (with Coulomb): forces = -dE/dx
    conf = md_test_config(kgrid=[2, 2, 2])
    strc = md_reference_structure(conf)
    pot = MDPotential(strc, conf, comm, electrons="exact", rep=md_test_repulsive(conf))
    x = displaced(pot.tb.reference, 0.1)
    an, fd = fd_force_check(pot, x)
    @test an ≈ fd rtol=1e-6
    res = md_compute!(pot, x)
    @test norm(sum(res.forces, dims=2)) < 1e-8          # no net force

    # SOC
    conf = md_test_config(soc=true, kgrid=[1, 1, 2])
    pot = MDPotential(md_reference_structure(conf), conf, comm, electrons="exact")
    an, fd = fd_force_check(pot, displaced(pot.tb.reference, 0.1))
    @test an ≈ fd rtol=1e-6

    # KPM against the exact solver at Γ: with a probing radius larger than the cell, every atom has its own color and
    # the density matrix is exact up to the Chebyshev expansion
    conf = md_test_config(supercell=[2, 2, 2])
    set_value!(conf, "probe_radius", "MD", 30.0)
    set_value!(conf, "kpm_tol", "MD", 1e-9)
    strc = md_reference_structure(conf)
    pe = MDPotential(strc, conf, comm, electrons="exact")
    pk = MDPotential(strc, conf, comm, electrons="kpm")
    x = displaced(pe.tb.reference, 0.1)
    re = md_compute!(pe, x); rk = md_compute!(pk, x)
    @test pk.solver.nvec == pe.pat.N
    @test rk.E ≈ re.E atol=1e-6
    @test maximum(abs, rk.forces - re.forces) < 1e-5
    @test rk.w ≈ re.w atol=1e-6
    # second call keeps μ (one Chebyshev pass)
    md_compute!(pk, x)
    @test pk.solver.npasses == 1
    # With probing (fewer vectors than orbitals), the error decreases with the probing radius
    errs = map((5.0, 9.0)) do radius
        set_value!(conf, "probe_radius", "MD", radius)
        pr = MDPotential(strc, conf, comm, electrons="kpm")
        @test pr.solver.nvec < pr.pat.N
        maximum(abs, md_compute!(pr, x).forces - re.forces)
    end
    @test errs[2] < errs[1]
    # single precision
    set_value!(conf, "kpm_precision", "MD", "single")
    set_value!(conf, "probe_radius", "MD", 30.0)
    ps = MDPotential(strc, conf, comm, electrons="kpm")
    @test md_compute!(ps, x).E ≈ re.E rtol=1e-5

    # Device path (KernelAbstractions kernels) with JLArrays
    set_value!(conf, "kpm_precision", "MD", "double")
    pd = MDPotential(strc, conf, comm, electrons="kpm", device=ArrayDevice(JLArray))
    rd = md_compute!(pd, x)
    @test rd.E ≈ rk.E atol=1e-8
    @test maximum(abs, rd.forces - rk.forces) < 1e-7
end

@testset "MD fit" begin
    # Synthetic data from a known nuclear term are fitted exactly
    mktempdir() do dir
        cd(dir) do
            conf = md_test_config()
            strc = md_reference_structure(conf)
            rep_true = md_test_repulsive(conf)
            pot = MDPotential(strc, conf, comm, electrons="exact", rep=rep_true)
            Nf = 12
            x0 = pot.tb.reference
            pos = stack([displaced(x0, 0.12, seed=n) for n in 1:Nf])
            res = [md_compute!(pot, pos[:, :, n]) for n in 1:Nf]
            data = MDH.MDData(pos, repeat(Matrix(pot.tb.lattice), 1, 1, Nf), pot.tb.types, [r.E for r in res],
                              stack([r.forces for r in res]), ["synthetic" for _ in 1:Nf])
            write_md_data("train.h5", MDH.MDData(pos[:, :, 1:8], data.lattice[:, :, 1:8], data.types, data.energies[1:8],
                                                 data.forces[:, :, 1:8], data.source[1:8]))
            write_md_data("val.h5", MDH.MDData(pos[:, :, 9:12], data.lattice[:, :, 9:12], data.types, data.energies[9:12],
                                               data.forces[:, :, 9:12], data.source[9:12]))
            @test read_md_data("train.h5").energies ≈ data.energies[1:8]
            set_value!(conf, "task", "MD", "fit")
            set_value!(conf, "train_data", "MD", "train.h5"); set_value!(conf, "val_data", "MD", "val.h5")
            set_value!(conf, "rep_rmin", "MD", 2.2); set_value!(conf, "lambda", "MD", 1e-12); set_value!(conf, "smoothness", "MD", 0.0)
            set_value!(conf, "electrons", "MD", "exact")
            out = Hamster.run_calculation(Val{:md}(), comm, conf)
            @test out.errors["train_rmse_F"] < 1e-6
            @test out.errors["val_rmse_F"] < 1e-6
            @test out.errors["val_rmse_E"] < 1e-6
            # pair potentials and Coulomb scale (the splines are only determined in the range of the data)
            rep = MDH.load_repulsive("md_model.h5")
            @test rep.coeffs[end] ≈ rep_true.coeffs[end] rtol=1e-3
            @test MDH.pair_energy(rep, 1, 4.2)[2] ≈ MDH.pair_energy(rep_true, 1, 4.2)[2] rtol=1e-3
            # The electronic part comes from the cache the second time
            @test isfile("md_electrons.h5")
            out2 = Hamster.run_calculation(Val{:md}(), comm, conf)
            @test out2.errors["val_rmse_F"] ≈ out.errors["val_rmse_F"]
            # eval
            set_value!(conf, "task", "MD", "eval"); set_value!(conf, "train_data", "MD", "val.h5")
            ev = Hamster.run_calculation(Val{:md}(), comm, conf)
            @test ev.errors["rmse_F"] < 1e-6
        end
    end
end

@testset "MD dynamics" begin
    mktempdir() do dir
        cd(dir) do
            conf = md_test_config(supercell=[2, 1, 1])
            MDH.save_repulsive(md_test_repulsive(conf), "md_model.h5")
            set_value!(conf, "task", "MD", "run")
            set_value!(conf, "electrons", "MD", "exact")
            set_value!(conf, "dt", "MD", 2.0); set_value!(conf, "temperature", "MD", 300.0)
            set_value!(conf, "traj_every", "MD", 2); set_value!(conf, "eig_every", "MD", 5)
            set_value!(conf, "rdf_rmax", "MD", 5.0)

            # NVE: the energy is conserved
            set_value!(conf, "thermostat", "MD", "none"); set_value!(conf, "nsteps", "MD", 100)
            set_value!(conf, "output", "MD", "nve.h5")
            Hamster.run_calculation(Val{:md}(), comm, conf)
            Econs = h5read("nve.h5", "thermo/E_cons"); Epot = h5read("nve.h5", "thermo/E_pot")
            @test std(Econs) < 0.02 * std(Epot)
            @test "analysis" in h5open(keys, "nve.h5")
            @test size(h5read("nve.h5", "eigenvalues/values"), 3) == 21
            @test size(h5read("nve.h5", "positions"), 3) == 51

            # CSVR: restart gives the same trajectory as an uninterrupted run
            set_value!(conf, "thermostat", "MD", "csvr"); set_value!(conf, "tau", "MD", 20.0)
            set_value!(conf, "nsteps", "MD", 40); set_value!(conf, "output", "MD", "full.h5")
            full = Hamster.run_calculation(Val{:md}(), comm, conf)
            set_value!(conf, "nsteps", "MD", 20); set_value!(conf, "output", "MD", "part.h5")
            Hamster.run_calculation(Val{:md}(), comm, conf)
            set_value!(conf, "nsteps", "MD", 40); set_value!(conf, "restart", "MD", true)
            part = Hamster.run_calculation(Val{:md}(), comm, conf)
            @test part.positions ≈ full.positions atol=1e-8
            @test h5read("part.h5", "thermo/E_cons") ≈ h5read("full.h5", "thermo/E_cons") atol=1e-7
            @test h5read("part.h5", "thermo/step") == 0:40

            # The analysis functions
            a = md_analysis("full.h5")
            @test all(isfinite, a.rdf["total"])
            @test sum(a.vdos["total"]) * (a.nu[2] - a.nu[1]) ≈ 3 * 10 rtol=1e-8
        end
    end
end

@testset "MD thermostat" begin
    # CSVR samples the canonical distribution of the kinetic energy of free particles
    m = fill(50.0, 100)
    v = MDH.maxwell_boltzmann_velocities(m, 400.0, 1)
    @test MDH.temperature(MDH.kinetic_energy(v, m), 297) ≈ 400.0
    Ts = Float64[]
    for step in 1:4000
        MDH.csvr_rescale!(v, m, 297, 300.0, 10.0, 50.0, MDH.step_rng(1, step))
        push!(Ts, MDH.temperature(MDH.kinetic_energy(v, m), 297))
    end
    @test mean(Ts[1000:end]) ≈ 300.0 rtol=0.03
    @test std(Ts[1000:end]) ≈ 300.0 * sqrt(2 / 297) rtol=0.2
end
