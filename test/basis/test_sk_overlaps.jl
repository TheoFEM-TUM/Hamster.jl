# Slater-Koster convention: the geometry tensor with all parameters equal to 1 is the overlap integral of the orbitals,
# for every pair of orbitals and bond direction. Ga-As dimer with s and p orbitals on both atoms (the case in which an
# overlap of different elements contributes only in its own order of the angular momenta, e.g. Ga+As_spσ: s on Ga).
function sp_dimer_config(dir)
    # Ga at the origin and As along a general direction (no periodic images within rcut)
    write(joinpath(dir, "POSCAR"), """
        GaAs dimer
        1.0
        12.0 0 0
        0 12.0 0
        0 0 12.0
        Ga As
        1 1
        Direct
        0.0 0.0 0.0
        0.1083333333 0.0750000000 0.1416666667
        """)
    conf = get_empty_config()
    set_value!(conf, "poscar", joinpath(dir, "POSCAR")); set_value!(conf, "rcut", 4.0); set_value!(conf, "verbosity", 0)
    set_value!(conf, "init_params", "ones"); set_value!(conf, "rllm_file", joinpath(dir, "rllm.dat"))
    for (element, alpha) in (("Ga", 14.3), ("As", 18))
        set_value!(conf, "orbitals", element, "s px py pz"); set_value!(conf, "n", element, 4); set_value!(conf, "alpha", element, alpha)
    end
    return conf
end

@testset "Geometry tensor = overlap integrals (s, p on different elements)" begin
    mktempdir() do dir
        conf = sp_dimer_config(dir)
        strc = Structure(conf); basis = Basis(strc, conf)
        Hamster.get_rllm(basis.overlaps, conf)
        orbs = [(iion, o) for iion in eachindex(basis.orbitals) for o in basis.orbitals[iion]]
        radial(iion) = (el = Hamster.number_to_element(strc.ions[iion].type); Hamster.ndict[Hamster.get_n(conf, el)](Hamster.get_alpha(conf, el)))
        function true_overlap(a, b)
            (ia, oa), (ib, ob) = orbs[a], orbs[b]
            r_a, r_b = strc.ions[ia].pos, strc.ions[ib].pos
            f = Hamster.Overlap(oa.type, radial(ia), r_a, ob.type, radial(ib), r_b)
            return hcubature(f, min.(r_a, r_b) .- 6, max.(r_a, r_b) .+ 6, rtol=1e-5, atol=1e-7, maxevals=2_000_000)[1]
        end
        pairs = [(a, b) for a in eachindex(orbs), b in eachindex(orbs) if orbs[a][1] ≠ orbs[b][1]]
        h = Hamster.get_geometry_tensor(strc, basis, conf)
        R0 = findfirst(r -> iszero(strc.Rs[:, r]), axes(strc.Rs, 2))
        G = sum(Matrix(h[v, R0]) for v in axes(h, 1))
        reference = [true_overlap(p...) for p in pairs]
        @test maximum(abs, reference) > 0.1
        @test maximum(abs(G[p...] - t) for (p, t) in zip(pairs, reference)) < 2e-3      # interpolation accuracy of the rllm
    end
end

@testset "Orbital lists of the p orbitals" begin
    for p in (Hamster.px(), Hamster.py(), Hamster.pz())
        @test Hamster.get_orbital_list(p) == Hamster.Angular[Hamster.px(), Hamster.py(), Hamster.pz()]
    end
end
