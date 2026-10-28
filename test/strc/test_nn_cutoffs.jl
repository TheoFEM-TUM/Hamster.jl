using LinearAlgebra

# Deterministic pseudo-random displacements (standard deviation about σ / √2)
rattle(σ, dims; phase=1.7) = σ .* sin.(phase .* reshape(1:prod(dims), dims) .^ 2)

# Diamond Si: 8-atom conventional cell repeated n × n × n times (Cartesian positions, lattice)
function si_diamond(n=2; a=5.431)
    basis = [0 0 0; 0.5 0.5 0; 0.5 0 0.5; 0 0.5 0.5; 0.25 0.25 0.25; 0.75 0.75 0.25; 0.75 0.25 0.75; 0.25 0.75 0.75]
    fr = reduce(hcat, [(collect(r) .+ [i, j, k]) ./ n for i in 0:n-1, j in 0:n-1, k in 0:n-1 for r in eachrow(basis)])
    lattice = n * a * Matrix(1.0I, 3, 3)
    return lattice * fr, lattice
end

# Labels of all pairs within rcut, as get_geometry_tensor assigns them (distance with the distortion, cutoffs or
# thresholds of the reference positions), together with the undistorted distances
function pair_labels(strc, conf, nn_dict)
    Ts = Hamster.frac_to_cart(strc.Rs, strc.lattice)
    r_refs = Float64[]; labels = Int64[]
    for (i, j, R) in Hamster.iterate_nn_grid_points(strc.point_grid)
        r_ref = Hamster.normdiff(strc.ions[i].pos, strc.ions[j].pos, Ts[:, R])
        r_ref ≤ Hamster.get_rcut(conf) || continue
        r = Hamster.normdiff(strc.ions[i].pos - strc.ions[i].dist, Hamster.apply_pbc(strc.ions[j].pos, Ts[:, R]) - strc.ions[j].dist)
        push!(r_refs, r_ref)
        push!(labels, Hamster.get_nn_label(r, nn_dict[Hamster.IonLabel(strc.ions[i].type, strc.ions[j].type, sorted=false)], conf))
    end
    return r_refs, labels
end

cutoffs(strc, conf) = Hamster.get_nn_cutoffs(strc.ions, Hamster.frac_to_cart(strc.Rs, strc.lattice), strc.point_grid, conf)
thresholds(strc, conf) = Hamster.get_nn_thresholds(strc.ions, Hamster.frac_to_cart(strc.Rs, strc.lattice), strc.point_grid, conf)

function sepnn_conf(rcut)
    conf = get_empty_config()
    set_value!(conf, "rcut", rcut); set_value!(conf, "sepNN", true); set_value!(conf, "verbosity", 0)
    return conf
end

@testset "Nearest-neighbor cutoffs (sepNN)" begin
    test_file_path = joinpath(@__DIR__, "test_files")

    # Diamond Si (64 atoms): midpoint between the first (a√3/4) and second (a/√2) shell
    conf = sepnn_conf(5.0)
    rs, lattice = si_diamond(2)
    types = fill(Hamster.element_to_number("Si"), size(rs, 2))
    Rs = Hamster.get_translation_vectors(rs, lattice, rcut=5.0)
    si = Structure(Rs, rs, zeros(size(rs)), types, lattice, conf)
    @test only(unique(values(cutoffs(si, conf)))) ≈ (5.431 * sqrt(3) / 4 + 5.431 / sqrt(2)) / 2

    # Without sepNN: zeros, as get_nn_thresholds
    conf_nosep = sepnn_conf(5.0); set_value!(conf_nosep, "sepNN", false)
    @test cutoffs(si, conf_nosep) == thresholds(si, conf_nosep)

    # Undistorted structures: the same labels as with the nearest-neighbor thresholds (backward compatibility:
    # existing parameter files give the same Hamiltonians)
    gaas = Structure(sepnn_conf(5.0), poscar_path=joinpath(test_file_path, "SC_POSCAR_gaas"))
    cspbbr3 = Structure(sepnn_conf(6.0), poscar_path=joinpath(test_file_path, "POSCAR_CsPbBr3"))
    for (strc, c) in ((si, conf), (gaas, sepnn_conf(5.0)), (cspbbr3, sepnn_conf(6.0)))
        @test pair_labels(strc, c, cutoffs(strc, c))[2] == pair_labels(strc, c, thresholds(strc, c))[2]
    end

    # Distorted structures: every bond of the first shell of the reference keeps label 1, the farther shells label 2,
    # in MD mode (reference positions + displacements) and for a distorted POSCAR (no reference)
    nn_si = 5.431 * sqrt(3) / 4
    δ = rattle(0.07, size(rs))
    si_md = Structure(Rs, rs, δ, types, lattice, conf)
    si_pc = Structure(Rs, rs .- δ, zeros(size(rs)), types, lattice, conf)
    for strc in (si_md, si_pc)
        r_refs, labels = pair_labels(strc, conf, cutoffs(strc, conf))
        first_shell = abs.(r_refs .- nn_si) .< (strc === si_md ? 1e-8 : 0.5)
        @test count(first_shell) == 256                                       # 64 atoms × 4 bonds
        @test all(labels[first_shell] .== 1) && all(labels[.!first_shell .& (r_refs .> 0)] .== 2)        # 0: on-site
        # the nearest-neighbor thresholds label most of the stretched bonds as farther neighbors
        @test count(==(2), pair_labels(strc, conf, thresholds(strc, conf))[2][first_shell]) > 100
    end

    # Compound (GaAs supercell) with a deterministic distortion
    conf_gaas = sepnn_conf(5.0)
    rs_g = reduce(hcat, Hamster.get_ion_positions(gaas.ions))
    δ_g = 0.05 .* sin.(reshape(1:length(rs_g), size(rs_g)))
    types_g = [ion.type for ion in gaas.ions]
    for strc in (Structure(gaas.Rs, rs_g, δ_g, types_g, gaas.lattice, conf_gaas),
                 Structure(gaas.Rs, rs_g .- δ_g, zeros(size(rs_g)), types_g, gaas.lattice, conf_gaas))
        r_refs, labels = pair_labels(strc, conf_gaas, cutoffs(strc, conf_gaas))
        first_shell = abs.(r_refs .- 5.65 * sqrt(3) / 4) .< 0.3                   # Ga–As bonds (Ga–Ga, As–As: 4.0 Å)
        @test count(first_shell) == 32 && all(labels[first_shell] .== 1)
    end
end

@testset "Geometry tensor of distorted structures (sepNN)" begin
    # The geometry tensor (and so the Hamiltonian) changes continuously with the distortion: a tiny displacement
    # does not switch bonds to the parameters of the farther shells
    mktempdir() do dir
        conf = sepnn_conf(5.0)
        set_value!(conf, "orbitals", "Si", "s px py pz"); set_value!(conf, "n", "Si", 3); set_value!(conf, "alpha", "Si", 10)
        set_value!(conf, "rllm_file", joinpath(dir, "rllm.dat"))
        rs, lattice = si_diamond(1)
        types = fill(Hamster.element_to_number("Si"), size(rs, 2))
        Rs = Hamster.get_translation_vectors(rs, lattice, rcut=5.0)
        ideal = Structure(Rs, rs, zeros(size(rs)), types, lattice, conf)
        basis = Basis(ideal, conf)
        Hamster.get_rllm(basis.overlaps, conf)
        h0 = Hamster.get_geometry_tensor(ideal, basis, conf)
        δ = rattle(1e-4, size(rs), phase=2.3)
        for strc in (Structure(Rs, rs, δ, types, lattice, conf), Structure(Rs, rs .- δ, zeros(size(rs)), types, lattice, conf))
            h = Hamster.get_geometry_tensor(strc, basis, conf)
            @test size(h) == size(h0)
            @test maximum(abs(x) for (a, b) in zip(h, h0) for x in Matrix(a) - Matrix(b)) < 1e-2
            # label statistics: 8 atoms × 4 bonds / 2 nearest-neighbor bonds, all with label 1
            stats = Hamster.nn_label_statistics(strc, basis, conf)
            cutoff, n1, n2 = stats["Si+Si"]
            @test isapprox(cutoff, (5.431 * sqrt(3) / 4 + 5.431 / sqrt(2)) / 2, atol=1e-3) && n1 == 16 && n2 > 0
        end
        # TB model summary in hamster.out: the table of the labels
        cd(dir) do
            touch("hamster.out")
            strc = Structure(Rs, rs, δ, types, lattice, conf)
            TBModel([strc], [basis], comm, conf, label="training set")
            out = read("hamster.out", String)
            @test occursin("Nearest-neighbor labels (sepNN)", out)
            @test occursin(r"\n    Si\+Si\s+3\.096\s+16\s+\d+\n", out)
        end
    end
end
