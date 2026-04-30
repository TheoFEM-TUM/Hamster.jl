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
end