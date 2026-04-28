function brute_force_periodic_coulomb(
    pos,
    q,
    box;
    nimg::Int=3
)
    N = length(pos)

    phi = zeros(Float64, N)
    E = 0.0

    # image vectors n = (nx, ny, nz), with shift = box * n
    @views for i in 1:N
        ri = pos[i]

        for j in 1:N
            rj = pos[j]
            qj = q[j]

            for nx in -nimg:nimg, ny in -nimg:nimg, nz in -nimg:nimg
                # skip exact self interaction in the home cell
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

    # total energy = 1/2 sum_i q_i phi_i
    E = 0.5 * sum(q .* phi)

    return (
        potentials = phi,
        energy = E,
        nimg = nimg,
    )
end

@testset "Ewald" begin
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
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
    ewald = Hamster.ewald_sum(
        pos,
        q,
        box,
        strc.Rs,
        strc.point_grid,
        rcut=5.5,
        mesh_spacing=0.5,
        include_short=true,
        include_long=true,
    )

    # Test 1: test Ewald energy vs reference implementation
    latvecs = [v for v in eachcol(box)]
    sys = Ewalder.System(; latvecs=latvecs, pos=pos)
    E = Ewalder.energy(sys; charges=q)
    @test isapprox(ewald.energy, E * Hamster.ke, rtol=1e-3)

    # Test 2: test Ewald energy and potential vs brute force summation
    ewald2 = brute_force_periodic_coulomb(pos, q, box, nimg=7)
    @test isapprox(ewald.energy, ewald2.energy, rtol=1e-3)
    @test all(@. isapprox(ewald.potentials, ewald2.potentials, rtol=1e-2))
    @show ewald.potentials
end