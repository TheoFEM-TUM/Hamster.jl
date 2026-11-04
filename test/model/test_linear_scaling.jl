function write_gaas_supercell(file, n)
    cell = [((0, 0, 0), "Ga"), ((0, .5, .5), "Ga"), ((.5, 0, .5), "Ga"), ((.5, .5, 0), "Ga"),
            ((.25, .25, .25), "As"), ((.25, .75, .75), "As"), ((.75, .25, .75), "As"), ((.75, .75, .25), "As")]
    positions(element) = [(i + r[1], j + r[2], k + r[3]) ./ n for i in 0:n-1, j in 0:n-1, k in 0:n-1 for (r, e) in cell if e == element]
    ga, as = positions("Ga"), positions("As")
    open(file, "w") do f
        println(f, "GaAs supercell\n", 5.65 * n, "\n1 0 0\n0 1 0\n0 0 1\nGa As\n", length(ga), " ", length(as), "\nDirect")
        foreach(r -> println(f, join(r, " ")), vcat(ga, as))
    end
end

function growth(f, small, large)
    measure(x) = (f(x); minimum(@allocated(f(x)) for _ in 1:3))
    return measure(large) / measure(small)
end

@testset "Linear scaling with the number of atoms" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename=joinpath(path, "hconf_gaas"))
    set_value!(conf, "verbosity", 0)
    strcs = mktempdir() do dir
        map((4, 8)) do n
            file = joinpath(dir, "POSCAR_$n")
            write_gaas_supercell(file, n)
            Structure(conf, poscar_path=file)
        end
    end
    Nratio = length(strcs[2].ions) / length(strcs[1].ions)
    @test Nratio == 8

    @test growth(strc -> Hamster.get_orbitals(strc, conf), strcs...) < 1.25Nratio
    @test growth(strc -> count(_ -> true, Hamster.iterate_nn_grid_points(strc.point_grid)), strcs...) < 1.25Nratio
    @test growth(N -> Hamster.convert_block_matrix_to_sparse(BlockDiagonal([fill(1. + 0im, 8, 8) for _ in 1:N])), 1000, 8000) < 1.25Nratio
    @test growth(N -> Hamster.gradient_apply_spin_basis(spdiagm(0 => ones(N), 4 => ones(N - 4))), 4000, 32000) < 1.25Nratio
    Norb = fill(4, 2^17)
    Hamster.get_index_to_ion_orb_map(Norb)
    @test minimum(@elapsed(Hamster.get_index_to_ion_orb_map(Norb)) for _ in 1:3) < 0.1
end

@testset "Neighbor pairs of the point grid" begin
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename=joinpath(path, "hconf_gaas"))
    set_value!(conf, "verbosity", 0)
    for poscar in ("POSCAR_gaas", "POSCAR_cspbbr3")
        strc = Structure(conf, poscar_path=joinpath(path, poscar))
        pairs = collect(Hamster.iterate_nn_grid_points(strc.point_grid))
        @test allunique(pairs)
        reference = [(iion1, iion2, R) for (grid_point, inds) in strc.point_grid.dict0
                     for nn in Hamster.nn_grid_points(grid_point, strc.point_grid.dictR)
                     for iion1 in inds for (iion2, R) in strc.point_grid.dictR[nn]]
        @test pairs == reference
        @test length(pairs) == Hamster.count_nn_pairs(strc.point_grid)
        rs = Hamster.get_ion_positions(strc.ions); Ts = Hamster.frac_to_cart(strc.Rs, strc.lattice)
        within_rcut = Set((i, j, R) for i in eachindex(rs), j in eachindex(rs), R in axes(Ts, 2)
                          if Hamster.normdiff(rs[i], rs[j], Ts[:, R]) ≤ 5)
        @test within_rcut ⊆ Set(pairs)
        grid_point = first(keys(strc.point_grid.dict0))
        @test allunique(collect(Hamster.iterate_nn_grid_points(grid_point, strc.point_grid)))
    end
end

@testset "Index maps" begin
    Norb = [3, 0, 4, 1]
    index_map = Hamster.get_index_to_ion_orb_map(Norb)
    @test index_map == [Hamster.get_ion_and_orbital_indices(Norb, i) for i in 1:sum(Norb)]
    @test Hamster.get_ion_orb_to_index_map(Norb) == Dict(ion_orb => i for (i, ion_orb) in enumerate(index_map))
end

@testset "SOC gradient per species" begin
    path = joinpath(@__DIR__, "..", "soc", "test_files")
    conf = get_config(filename=joinpath(path, "hconf_cspbbr3"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_CsPbBr3"))
    set_value!(conf, "init_params", joinpath(path, "params_cspbbr3.dat"))
    set_value!(conf, "init_params", "ML", joinpath(path, "ml_params.dat"))
    set_value!(conf, "init_params", "SOC", joinpath(path, "params_cspbbr3.dat"))
    set_value!(conf, "update", "SOC", true)
    set_value!(conf, "verbosity", 0)
    strc = Structure(conf); basis = Basis(strc, conf)
    eff_ham = EffectiveHamiltonian([strc], [basis], comm, conf, rank=rank, nranks=nranks)
    soc_model = eff_ham.models[findfirst(m -> m isa Hamster.SOCModel, eff_ham.models)]
    @test length(soc_model.params) > 1

    Hr = Hamster.get_hr(soc_model, nothing, 1)
    Nε = size(Hr[1], 1)
    dL_dHr = [[sparse(ComplexF64[sin(i + 2j + R) + im * cos(3i - j) for i in 1:Nε, j in 1:Nε]) for R in eachindex(Hr)]]
    reg = Hamster.Regularization(0., 0., 1)
    reference = map(eachindex(soc_model.params)) do p
        Msoc = BlockDiagonal([(soc_model.param_labels[p] == type) .* soc_model.matrices[type] for type in soc_model.types_per_strc[1]])
        real(sum(dL_dHr[1][soc_model.Rs_info[2, 1]] .* Matrix(Msoc)))
    end
    @test Hamster.get_model_gradient(soc_model, [1], reg, dL_dHr) ≈ reference
end
