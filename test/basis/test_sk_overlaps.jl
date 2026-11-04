# Slater-Koster convention: the geometry tensor with all parameters equal to 1 is the overlap integral of the orbitals,
# for every pair of orbitals and bond direction. Dimers with the same orbitals on both atoms (s, p, d), in a general
# bond direction.
function sp_dimer_config(dir; elements=("Ga", "As"), orbitals="s px py pz", orbitals_2=orbitals)
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
    for (element, alpha, orbs) in (("Ga", 14.3, orbitals), ("As", 18, orbitals_2))
        set_value!(conf, "orbitals", element, orbs); set_value!(conf, "n", element, 4); set_value!(conf, "alpha", element, alpha)
    end
    return conf
end

# A hybrid sp3 orbital with axis a is s / 2 + √3 / 2 (a ⋅ p)
components(o) = o.type isa Hamster.sp3 ? [(Hamster.s(), 1 / 2); collect(zip((Hamster.px(), Hamster.py(), Hamster.pz()), √3 / 2 .* o.axis))] : [(o.type, 1.)]

function geometry_vs_overlap(conf)
    strc = Structure(conf); basis = Basis(strc, conf)
    Hamster.get_rllm(basis.overlaps, conf)
    orbs = [(iion, o) for iion in eachindex(basis.orbitals) for o in basis.orbitals[iion]]
    radial(iion) = (el = Hamster.number_to_element(strc.ions[iion].type); Hamster.ndict[Hamster.get_n(conf, el)](Hamster.get_alpha(conf, el)))
    function true_overlap(a, b)
        (ia, oa), (ib, ob) = orbs[a], orbs[b]
        r_a, r_b = strc.ions[ia].pos, strc.ions[ib].pos
        sum(components(oa)) do (ϕa, ca)
            sum(components(ob)) do (ϕb, cb)
                f = Hamster.Overlap(ϕa, radial(ia), r_a, ϕb, radial(ib), r_b)
                # integration box not symmetric around the atoms (a symmetric one gives false zeros for d orbitals)
                ca * cb * hcubature(f, min.(r_a, r_b) .- 6.137, max.(r_a, r_b) .+ 5.913, rtol=1e-4, atol=1e-6, maxevals=1_000_000)[1]
            end
        end
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

@testset "Geometry tensor = overlap integrals with hybrid orbitals" begin
    # hybrids next to hybrids, and next to s and p orbitals (frame of the angular functions, get_sign) on either atom
    for (elements, orbitals, orbitals_2) in ((("Ga", "Ga"), "sp3 sp3 sp3 sp3", "sp3 sp3 sp3 sp3"),
                                             (("Ga", "As"), "sp3 sp3 sp3 sp3", "sp3 sp3 sp3 sp3"),
                                             (("Ga", "As"), "sp3 sp3 sp3 sp3", "s px py pz"),
                                             (("Ga", "As"), "s px py pz", "sp3 sp3 sp3 sp3"))
        mktempdir() do dir
            largest, error = geometry_vs_overlap(sp_dimer_config(dir, elements=elements, orbitals=orbitals, orbitals_2=orbitals_2))
            @test largest > 0.1
            @test error < 2e-3
        end
    end
end

@testset "Rotation invariance with hybrid orbitals" begin
    # Γ-point eigenvalues of a distorted GaAs cell with sp3dr2 hybrids next to s, p and d orbitals, rotated rigidly
    rotation(a, u) = (u = normalize(u); cos(a) * I + sin(a) * [0 -u[3] u[2]; u[3] 0 -u[1]; -u[2] u[1] 0] + (1 - cos(a)) * u * u')
    lattice = 5.65 * [0.5 0.0 0.5; 0.5 0.5 0.0; 0.0 0.5 0.5]
    mktempdir() do dir
        function energies(lattice)
            open(joinpath(dir, "POSCAR"), "w") do f
                println(f, "GaAs\n1.0"); foreach(i -> println(f, join(lattice[:, i], " ")), 1:3)
                println(f, "Ga As\n1 1\nDirect\n0 0 0\n0.26 0.24 0.25")
            end
            conf = get_empty_config()
            set_value!(conf, "poscar", joinpath(dir, "POSCAR")); set_value!(conf, "rcut", 5.0); set_value!(conf, "verbosity", 0)
            set_value!(conf, "rllm_file", joinpath(dir, "rllm.dat")); set_value!(conf, "load_rllm", isfile(joinpath(dir, "rllm.dat")))
            set_value!(conf, "orbitals", "Ga", "sp3dr2 sp3dr2 sp3dr2 sp3dr2"); set_value!(conf, "nnaxes", "Ga", true)
            set_value!(conf, "orbitals", "As", "s px py pz dxy dyz dxz dz2 dx2_y2")
            set_value!(conf, "n", "Ga", 3); set_value!(conf, "n", "As", 3)
            strc = Structure(conf); basis = Basis(strc, conf)
            isfile(joinpath(dir, "rllm.dat")) || Hamster.get_rllm(basis.overlaps, conf)
            h = Hamster.get_geometry_tensor(strc, basis, conf)
            return eigvals(Hermitian(Matrix(sum((1 + 0.1sin(3v)) * h[v, R] for v in axes(h, 1), R in axes(h, 2)))))
        end
        E0 = energies(lattice)
        for (a, u) in ((0.7, [1., 2., 3.]), (π, [1., 0., 0.]), (2.1, [-1., 0.4, 0.2]))
            @test energies(rotation(a, u) * lattice) ≈ E0 atol=1e-6
        end
    end
end

@testset "Orbital lists of the p orbitals" begin
    for p in (Hamster.px(), Hamster.py(), Hamster.pz())
        @test Hamster.get_orbital_list(p) == Hamster.Angular[Hamster.px(), Hamster.py(), Hamster.pz()]
    end
end
