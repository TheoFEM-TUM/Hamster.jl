using LinearAlgebra

# rcut_pair_selection: on which positions (reference or current frame) the pair cutoff rcut is applied

# Two Si atoms in a large cubic cell. Reference positions `rs` (columns), distortions `δ`: the distorted positions
# are `rs .- δ`. Default: reference distance 5.6 Å, distorted distance 4.5 Å (a pair that is beyond rcut = 5.5 in
# the reference but within it in the frame).
function dimer(; r_ref=5.6, r_dist=4.5)
    lattice = 20.0 * Matrix(1.0I, 3, 3)
    rs = [zeros(3) [r_ref, 0.0, 0.0]]
    δ = [zeros(3) [r_ref - r_dist, 0.0, 0.0]]
    types = fill(Hamster.element_to_number("Si"), 2)
    return rs, δ, types, lattice
end

function dimer_conf(; rcut=5.5, rcut_tol=1.0, selection=nothing, grid_size=nothing)
    conf = get_empty_config()
    set_value!(conf, "rcut", rcut)
    set_value!(conf, "rcut_tol", rcut_tol)
    isnothing(selection) || set_value!(conf, "rcut_pair_selection", selection)
    isnothing(grid_size) || set_value!(conf, "grid_size", grid_size)
    set_value!(conf, "orbitals", "Si", "s")
    set_value!(conf, "n", "Si", 3)
    set_value!(conf, "alpha", "Si", 10)
    set_value!(conf, "verbosity", 0)
    return conf
end

# Structure with the Rs and grid that production (get_structures) would build: translations for rcut_search
function dimer_structure(conf; r_ref=5.6, r_dist=4.5)
    rs, δ, types, lattice = dimer(r_ref=r_ref, r_dist=r_dist)
    dmax = maximum(norm, eachcol(δ))
    Rs = Hamster.get_translation_vectors(rs, lattice, rcut=Hamster.rcut_search(conf, dmax))
    return Structure(Rs, rs, δ, types, lattice, conf)
end

nnz_total(h) = sum(nnz, h)

@testset "pair_in_cutoff" begin
    # reference mode keeps the current rule: reference distance within rcut, distorted distance below rcut + |tol|
    @test Hamster.pair_in_cutoff(5.4, 6.2, 5.5, 1.0, "reference") == true
    @test Hamster.pair_in_cutoff(5.6, 4.5, 5.5, 1.0, "reference") == false    # beyond rcut in the reference
    @test Hamster.pair_in_cutoff(5.4, 6.6, 5.5, 1.0, "reference") == false    # 6.6 - |1.0| ≥ rcut
    # frame mode: the distorted distance alone decides (support of the cutoff function)
    @test Hamster.pair_in_cutoff(5.6, 4.5, 5.5, 1.0, "frame") == true         # within rcut in the frame
    @test Hamster.pair_in_cutoff(5.4, 6.2, 5.5, 1.0, "frame") == true         # 6.2 < rcut + rcut_tol
    @test Hamster.pair_in_cutoff(5.4, 6.6, 5.5, 1.0, "frame") == false        # 6.6 ≥ rcut + rcut_tol
    # negative rcut_tol: frame mode cuts at rcut, reference mode allowed up to rcut + |rcut_tol|
    @test Hamster.pair_in_cutoff(5.4, 5.6, 5.5, -1.0, "reference") == true
    @test Hamster.pair_in_cutoff(5.4, 5.6, 5.5, -1.0, "frame") == false
    @test Hamster.pair_in_cutoff(5.6, 4.5, 5.5, -1.0, "reference") == false
    @test Hamster.pair_in_cutoff(5.6, 4.5, 5.5, -1.0, "frame") == true
end

@testset "rcut_pair_selection tag" begin
    conf = get_empty_config()
    @test Hamster.get_rcut_pair_selection(conf) == "reference"                 # default
    set_value!(conf, "rcut_pair_selection", "FRAME")
    @test Hamster.get_rcut_pair_selection(conf) == "frame"                     # case-insensitive
    mktempdir() do dir
        file = joinpath(dir, "hconf")
        write(file, "begin Options\n    rcut_pair_selection = frame\nend\n")
        @test_logs get_config(filename=file)                                   # valid tag: no warnings
        @test get_config(filename=file)("rcut_pair_selection") == "frame"
    end
    # the setting is written to the params.dat header and checked by check_consistency
    conf_ref = dimer_conf(selection="reference")
    @test Hamster.check_consistency(Dict("rcut_pair_selection" => "reference"), conf_ref)
    @test_logs (:warn, r"not consistent") Hamster.check_consistency(Dict("rcut_pair_selection" => "frame"), conf_ref) == false
    conf_frame = dimer_conf(selection="frame")
    @test Hamster.check_consistency(Dict("rcut_pair_selection" => "frame"), conf_frame)
    @test_logs (:warn, r"not consistent") Hamster.check_consistency(Dict("rcut_pair_selection" => "reference"), conf_frame) == false
end

@testset "rcut_search and max_displacement" begin
    conf_ref = dimer_conf(selection="reference")
    conf_frame = dimer_conf(selection="frame")
    @test Hamster.rcut_search(conf_ref, 1.1) == 5.5                            # reference mode: unchanged
    @test Hamster.rcut_search(conf_frame, 1.1) ≈ 5.5 + 1.0 + 2 * 1.1           # frame mode: rcut + rcut_tol + 2 dmax
    rs, δ, types, lattice = dimer(r_ref=5.6, r_dist=4.5)
    rs_all = cat(rs, rs .- δ, dims=3)                                          # frame 1 = reference
    @test Hamster.max_displacement(rs, rs_all, lattice) ≈ 1.1
end

@testset "rcut_pair_selection = reference reproduces the default bit for bit" begin
    mktempdir() do dir
        conf_def = dimer_conf()                                                # tag not set
        conf_ref = dimer_conf(selection="reference")                           # tag set explicitly
        set_value!(conf_def, "rllm_file", joinpath(dir, "rllm_def.dat"))
        set_value!(conf_ref, "rllm_file", joinpath(dir, "rllm_ref.dat"))
        strc_def = dimer_structure(conf_def)
        basis_def = Basis(strc_def, conf_def)
        Hamster.get_rllm(basis_def.overlaps, conf_def)
        h_def = Hamster.get_geometry_tensor(strc_def, basis_def, conf_def)

        strc_ref = dimer_structure(conf_ref)
        basis_ref = Basis(strc_ref, conf_ref)
        Hamster.get_rllm(basis_ref.overlaps, conf_ref)
        h_ref = Hamster.get_geometry_tensor(strc_ref, basis_ref, conf_ref)

        @test size(h_def) == size(h_ref)
        @test all(Matrix(a) == Matrix(b) for (a, b) in zip(h_def, h_ref))      # exactly identical
        @test strc_def.point_grid.grid_size == strc_ref.point_grid.grid_size == 5.5
        # the pair at 5.6 Å in the reference has no hopping (rcut = 5.5), although it is at 4.5 Å in the frame
        R0 = Hamster.findR0(strc_ref.Rs)
        @test nnz_total(h_ref) == 2 && all(iszero(h_ref[v, R0][1, 2]) for v in axes(h_ref, 1))
    end
end

@testset "rcut_pair_selection = frame includes the pair of the current geometry" begin
    mktempdir() do dir
        conf_frame = dimer_conf(selection="frame", grid_size=5.5)              # user grid_size = rcut
        set_value!(conf_frame, "rllm_file", joinpath(dir, "rllm.dat"))
        strc = dimer_structure(conf_frame)                                     # reference 5.6 Å, frame 4.5 Å
        # the neighbor grid is enlarged to cover the distorted pairs (reference distance up to 2 dmax farther)
        @test strc.point_grid.grid_size ≈ Hamster.rcut_search(conf_frame, 1.1)
        basis = Basis(strc, conf_frame)
        Hamster.get_rllm(basis.overlaps, conf_frame)
        h = Hamster.get_geometry_tensor(strc, basis, conf_frame)
        R0 = Hamster.findR0(strc.Rs)
        @test nnz_total(h) > 2
        @test any(!iszero(h[v, R0][1, 2]) for v in axes(h, 1))                # hopping of the 4.5 Å pair
        # the sparse iterator of the Hamiltonian uses the same pair set
        it = Hamster.get_sparse_iterator(strc, basis, conf_frame)
        @test any(==((1, 2)), it[R0])
        # bonds agree
        bonds = Hamster.get_bonds(strc, basis, conf_frame)
        @test any(!iszero, bonds[R0][1][1, 2])
        # the ML descriptor uses the same pair set
        model = TBModel(strc, basis, conf_frame)
        d_frame = Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, conf_frame)
        @test nnz(d_frame[R0]) > 0
        conf_ref = dimer_conf(selection="reference")
        set_value!(conf_ref, "rllm_file", joinpath(dir, "rllm_ref2.dat"))
        strc_ref = dimer_structure(conf_ref)
        basis_ref = Basis(strc_ref, conf_ref)
        Hamster.get_rllm(basis_ref.overlaps, conf_ref)
        model_ref = TBModel(strc_ref, basis_ref, conf_ref)
        d_ref = Hamster.get_tb_descriptor(model_ref.hs, model_ref.params, strc_ref, basis_ref, conf_ref)
        @test nnz(d_frame[R0]) > nnz(d_ref[R0])
        # label statistics see the pair as well
        stats = Hamster.nn_label_statistics(strc, basis, conf_frame)
        @test stats["Si+Si"][2] + stats["Si+Si"][3] > 0
    end
end

@testset "rcut_tol defaults to 0 in frame mode" begin
    conf = get_empty_config()
    @test Hamster.get_rcut_tol(conf) == 1.0                                     # reference (default): unchanged
    set_value!(conf, "rcut_pair_selection", "frame")
    @test Hamster.get_rcut_tol(conf) == 0.0                                     # frame: hard cutoff at rcut
    set_value!(conf, "rcut_tol", 2.0)
    @test Hamster.get_rcut_tol(conf) == 2.0                                     # explicit value wins
    set_value!(conf, "rcut_pair_selection", "reference")
    @test Hamster.get_rcut_tol(conf) == 2.0
end

@testset "missed_bond_statistics and missed-bonds warning" begin
    # dimer with reference distance 5.6 Å (beyond rcut = 5.5) and distorted distance 4.5 Å: one missed bond
    conf_ref = dimer_conf(selection="reference")
    strc = dimer_structure(conf_ref)
    @test Hamster.missed_bond_statistics(strc, conf_ref) == Dict("Si+Si" => 1.0)
    # the count does not depend on the pair selection (only the emitted warning does)
    conf_frame = dimer_conf(selection="frame")
    @test Hamster.missed_bond_statistics(dimer_structure(conf_frame), conf_frame) == Dict("Si+Si" => 1.0)
    # a pair within rcut in the reference is not missed
    conf_in = dimer_conf(selection="reference")
    @test isempty(Hamster.missed_bond_statistics(dimer_structure(conf_in, r_ref=5.4, r_dist=4.5), conf_in))
    # end to end: get_structures warns in reference mode (small rcut: Ga-As bonds of the XDATCAR are missed)
    path = joinpath(@__DIR__, "test_files")
    conf_md = get_empty_config()
    set_value!(conf_md, "rcut", 2.0)
    set_value!(conf_md, "verbosity", 0)
    set_value!(conf_md, "XDATCAR", "Supercell", joinpath(path, "XDATCAR_gaas"))
    set_value!(conf_md, "POSCAR", "Supercell", joinpath(path, "SC_POSCAR_gaas"))
    @test_logs (:warn, r"Missed bonds") get_structures(conf_md, mode="md", config_indices=[1, 2, 3])
end

@testset "frame requires ML distortion flags" begin
    conf = dimer_conf(selection="frame")
    set_value!(conf, "model", "ML", true)                                      # ML block present, flags default to false
    @test_throws ErrorException dimer_structure(conf)
    set_value!(conf, "apply_distance_distortion", "ML", true)
    @test_throws ErrorException dimer_structure(conf)                         # apply_distortion still false
    set_value!(conf, "apply_distortion", "ML", true)
    dimer_structure(conf)                                                       # both true: no error
    # pure TB run (no ML block): no error
    dimer_structure(dimer_conf(selection="frame"))
end

@testset "rcut_pair_selection = frame: H independent of the reference" begin
    mktempdir() do dir
        # Same distorted geometry (4.5 Å), two different references: 5.4 Å (within rcut) and 6.5 Å (beyond rcut)
        conf_frame = dimer_conf(selection="frame")
        conf_ref = dimer_conf(selection="reference")
        hs_frame, hs_ref = [], []
        for r_ref in (5.4, 6.5)
            for (conf, sel) in ((conf_frame, "frame"), (conf_ref, "reference"))
                c = deepcopy(conf)
                set_value!(c, "rllm_file", joinpath(dir, "rllm_$(r_ref)_$sel.dat"))
                strc = dimer_structure(c, r_ref=r_ref, r_dist=4.5)
                basis = Basis(strc, c)
                Hamster.get_rllm(basis.overlaps, c)
                push!(sel == "frame" ? hs_frame : hs_ref, Hamster.get_geometry_tensor(strc, basis, c))
            end
        end
        h_frame_54, h_frame_65 = hs_frame
        h_ref_54, h_ref_65 = hs_ref
        # frame mode: same H for both references
        @test size(h_frame_54) == size(h_frame_65)
        @test all(Matrix(a) == Matrix(b) for (a, b) in zip(h_frame_54, h_frame_65))
        # reference mode: different H (the 5.4 Å pair has a hopping, the 6.5 Å pair has none)
        @test nnz_total(h_ref_54) > nnz_total(h_ref_65)
    end
end
