# Slater-Koster convention: the geometry tensor with all parameters equal to 1 is the overlap integral of the orbitals,
# for every pair of orbitals and bond direction. Dimers with the same orbitals on both atoms (s, p, d), in a general
# bond direction.
function sp_dimer_config(dir; elements=("Ga", "As"), orbitals="s px py pz")
    # Atom 1 at the origin and atom 2 along a general direction (no periodic images within rcut)
    species = elements[1] == elements[2] ? "$(elements[1])\n2" : "$(elements[1]) $(elements[2])\n1 1"
    write(joinpath(dir, "POSCAR"), """
        dimer
        1.0
        12.0 0 0
        0 12.0 0
        0 0 12.0
        $species
        Direct
        0.0 0.0 0.0
        0.1083333333 0.0750000000 0.1416666667
        """)
    conf = get_empty_config()
    set_value!(conf, "poscar", joinpath(dir, "POSCAR")); set_value!(conf, "rcut", 4.0); set_value!(conf, "verbosity", 0)
    set_value!(conf, "init_params", "ones"); set_value!(conf, "rllm_file", joinpath(dir, "rllm.dat"))
    for (element, alpha) in (("Ga", 14.3), ("As", 18))
        set_value!(conf, "orbitals", element, orbitals); set_value!(conf, "n", element, 4); set_value!(conf, "alpha", element, alpha)
    end
    return conf
end

function geometry_vs_overlap(conf)
    strc = Structure(conf); basis = Basis(strc, conf)
    Hamster.get_rllm(basis.overlaps, conf)
    orbs = [(iion, o) for iion in eachindex(basis.orbitals) for o in basis.orbitals[iion]]
    radial(iion) = (el = Hamster.number_to_element(strc.ions[iion].type); Hamster.ndict[Hamster.get_n(conf, el)](Hamster.get_alpha(conf, el)))
    function true_overlap(a, b)
        (ia, oa), (ib, ob) = orbs[a], orbs[b]
        r_a, r_b = strc.ions[ia].pos, strc.ions[ib].pos
        f = Hamster.Overlap(oa.type, radial(ia), r_a, ob.type, radial(ib), r_b)
        # integration box not symmetric around the atoms (a symmetric one gives false zeros for d orbitals)
        return hcubature(f, min.(r_a, r_b) .- 6.137, max.(r_a, r_b) .+ 5.913, rtol=1e-4, atol=1e-6, maxevals=1_000_000)[1]
    end
    pairs = [(a, b) for a in eachindex(orbs), b in eachindex(orbs) if orbs[a][1] ≠ orbs[b][1]]
    h = Hamster.get_geometry_tensor(strc, basis, conf)
    R0 = findfirst(r -> iszero(strc.Rs[:, r]), axes(strc.Rs, 2))
    G = sum(Matrix(h[v, R0]) for v in axes(h, 1))
    reference = [true_overlap(p...) for p in pairs]
    return maximum(abs, reference), maximum(abs(G[p...] - t) for (p, t) in zip(pairs, reference))
end

@testset "Geometry tensor = overlap integrals" begin
    # s, p on different elements: an overlap contributes only in its own order of the angular momenta
    # s, p, d on different elements and on the same element: signs of the reversed orders, d angular functions
    for (elements, orbitals) in ((("Ga", "As"), "s px py pz"), (("Ga", "As"), "s px py pz dxy dyz dxz dz2 dx2_y2"),
                                 (("Ga", "Ga"), "s px py pz dxy dyz dxz dz2 dx2_y2"))
        mktempdir() do dir
            largest, error = geometry_vs_overlap(sp_dimer_config(dir, elements=elements, orbitals=orbitals))
            @test largest > 0.1
            @test error < 2e-3                                                      # interpolation accuracy of the rllm
        end
    end
end

@testset "Orbital lists of the p orbitals" begin
    for p in (Hamster.px(), Hamster.py(), Hamster.pz())
        @test Hamster.get_orbital_list(p) == Hamster.Angular[Hamster.px(), Hamster.py(), Hamster.pz()]
    end
end
