using LinearAlgebra

# rcut_pair_selection: on which positions (reference or current frame) the pair cutoff rcut is applied

# Two Si atoms in a large cubic cell. Reference positions `rs` (columns), distortions `δ`: the distorted positions
# are `rs .- δ`. Default: reference distance 5.6 Å, distorted distance 4.5 Å (a pair that is beyond rcut = 5.5 in
# the reference but within it in the frame).
function dimer(; r_ref=5.6, r_dist=4.5, a=20.0)
    lattice = a * Matrix(1.0I, 3, 3)
    rs = [zeros(3) [r_ref, 0.0, 0.0]]
    δ = [zeros(3) [r_ref - r_dist, 0.0, 0.0]]
    types = fill(Hamster.element_to_number("Si"), 2)
    return rs, δ, types, lattice
end

function dimer_conf(; rcut=5.5, rcut_tol=1.0, selection=nothing, grid_size=nothing, sepNN=false)
    conf = get_empty_config()
    set_value!(conf, "rcut", rcut)
    set_value!(conf, "rcut_tol", rcut_tol)
    isnothing(selection) || set_value!(conf, "rcut_pair_selection", selection)
    isnothing(grid_size) || set_value!(conf, "grid_size", grid_size)
    sepNN && set_value!(conf, "sepNN", true)
    set_value!(conf, "orbitals", "Si", "s")
    set_value!(conf, "n", "Si", 3)
    set_value!(conf, "alpha", "Si", 10)
    set_value!(conf, "verbosity", 0)
    return conf
end

# Structure with the Rs and grid that production (get_structures) would build: translations for rcut_search
function dimer_structure(conf; r_ref=5.6, r_dist=4.5, a=20.0)
    rs, δ, types, lattice = dimer(r_ref=r_ref, r_dist=r_dist, a=a)
    dmax = maximum(norm, eachcol(δ))
    Rs = Hamster.get_translation_vectors(rs, lattice, rcut=Hamster.rcut_search(conf, dmax))
    return Structure(Rs, rs, δ, types, lattice, conf)
end

function dimer_model(strc, conf, dir, name)
    c = deepcopy(conf)
    set_value!(c, "rllm_file", joinpath(dir, "rllm_$name.dat"))
    basis = Basis(strc, c)
    Hamster.get_rllm(basis.overlaps, c)
    return c, basis, TBModel(strc, basis, c)
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
    @test Hamster.get_rcut_tol(conf) == 1.0                                    # rcut_tol default is 1 in both modes
    set_value!(conf, "rcut_pair_selection", "frame")
    @test Hamster.get_rcut_tol(conf) == 1.0
    set_value!(conf, "rcut_tol", 2.0)
    @test Hamster.get_rcut_tol(conf) == 2.0                                    # explicit value wins
    mktempdir() do dir
        file = joinpath(dir, "hconf")
        write(file, "begin Options\n    rcut_pair_selection = frame\nend\n")
        @test_logs get_config(filename=file)                                   # valid tag: no warnings
        @test get_config(filename=file)("rcut_pair_selection") == "frame"
    end
    # the setting is written to the params.dat header and checked by check_consistency; a missing key in an old
    # params.dat counts as "reference"
    conf_ref = dimer_conf(selection="reference")
    conf_frame = dimer_conf(selection="frame")
    @test Hamster.check_consistency(Dict("rcut_pair_selection" => "reference"), conf_ref)
    @test Hamster.check_consistency(Dict("rcut_pair_selection" => "frame"), conf_frame)
    @test_logs (:warn, r"not consistent") Hamster.check_consistency(Dict("rcut_pair_selection" => "frame"), conf_ref) == false
    @test_logs (:warn, r"not consistent") Hamster.check_consistency(Dict("rcut_pair_selection" => "reference"), conf_frame) == false
    @test Hamster.check_consistency(Dict{String,String}(), conf_ref)           # old file without the key: reference
    @test_logs (:warn, r"not consistent") Hamster.check_consistency(Dict{String,String}(), conf_frame) == false
end

@testset "ml_params.dat header stores rcut_pair_selection" begin
    mktempdir() do dir
        file = joinpath(dir, "ml_params.dat")
        write(file, "begin unknown\n  rcut_pair_selection = frame\nend\n")
        conf_ref = dimer_conf(selection="reference")
        conf_frame = dimer_conf(selection="frame")
        @test_throws ErrorException Hamster.check_ml_settings(conf_ref, file)
        @test Hamster.check_ml_settings(conf_frame, file) === nothing
        # apply_ml_settings! takes the value from the file if the Options block does not set it
        conf_unset = dimer_conf()
        set_value!(conf_unset, "init_params", "ML", file)
        Hamster.apply_ml_settings!(conf_unset)
        @test Hamster.get_rcut_pair_selection(conf_unset) == "frame"
    end
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

@testset "missed_bond_statistics and missed-bonds warning" begin
    conf_ref = dimer_conf(selection="reference")
    # dimer with reference distance 5.6 Å (beyond rcut = 5.5) and distorted distance 4.5 Å: one missed bond
    strc = dimer_structure(conf_ref)
    @test Hamster.missed_bond_statistics(strc, conf_ref) == Dict("Si+Si" => 1.0)
    # the count does not depend on the pair selection (only the emitted warning does)
    conf_frame = dimer_conf(selection="frame")
    @test Hamster.missed_bond_statistics(dimer_structure(conf_frame), conf_frame) == Dict("Si+Si" => 1.0)
    # a pair within rcut in the reference is not missed
    @test isempty(Hamster.missed_bond_statistics(dimer_structure(conf_ref, r_ref=5.4, r_dist=4.5), conf_ref))
    # position-robust: reference distance 12 Å spans more than one grid cell of the structure's own grid
    # (cell 5.5 in reference mode), the diagnostic search must still find it
    strc_far = dimer_structure(conf_ref, r_ref=12.0, r_dist=4.5)
    @test Hamster.missed_bond_statistics(strc_far, conf_ref) == Dict("Si+Si" => 1.0)
    # the warning aggregates over structures and is emitted once per process (on MPI rank 0)
    if !MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0
        Hamster.MISSED_BONDS_WARNED[] = false
        @test_logs (:warn, r"Missed bonds.*Si\+Si: 2\.0") Hamster.warn_missed_bonds([strc, strc], conf_ref)
        @test_logs Hamster.warn_missed_bonds([strc], conf_ref)                 # only once
        @test_logs Hamster.warn_missed_bonds([strc], conf_frame)               # not in frame mode
    end
end

@testset "grid_size enlarged warning only for an explicit grid_size" begin
    conf_silent = dimer_conf(selection="frame")                                # grid_size not set: silent
    @test_logs dimer_structure(conf_silent)
    @test dimer_structure(conf_silent).point_grid.grid_size ≈ 8.7
    conf_explicit = dimer_conf(selection="frame", grid_size=5.5)
    @test_logs (:warn, r"enlarged to") dimer_structure(conf_explicit)
end

@testset "caller-passed Rs coverage warning in frame mode" begin
    conf_frame = dimer_conf(selection="frame")
    rs, δ, types, lattice = dimer(r_ref=5.6, r_dist=4.5, a=12.0)                # images at 12-5.6=6.4 Å are needed
    dmax = 1.1
    Rs_full = Hamster.get_translation_vectors(rs, lattice, rcut=Hamster.rcut_search(conf_frame, dmax))
    @test_logs Hamster.warn_if_Rs_too_small(Rs_full, rs, lattice, conf_frame, dmax)   # complete: no warning
    @test_logs (:warn, r"may be missed") Hamster.warn_if_Rs_too_small(zeros(3, 1), rs, lattice, conf_frame, dmax)
    conf_ref = dimer_conf(selection="reference")
    @test_logs Hamster.warn_if_Rs_too_small(zeros(3, 1), rs, lattice, conf_ref, dmax)  # reference: no check
end

@testset "rcut_pair_selection = reference reproduces the default bit for bit" begin
    mktempdir() do dir
        conf_def = dimer_conf()                                                # tag not set
        conf_ref = dimer_conf(selection="reference")                           # tag set explicitly
        strc_def = dimer_structure(conf_def)
        c_def, basis_def, _ = dimer_model(strc_def, conf_def, dir, "def")
        h_def = Hamster.get_geometry_tensor(strc_def, basis_def, c_def)

        strc_ref = dimer_structure(conf_ref)
        c_ref, basis_ref, _ = dimer_model(strc_ref, conf_ref, dir, "ref")
        h_ref = Hamster.get_geometry_tensor(strc_ref, basis_ref, c_ref)

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
        strc = dimer_structure(conf_frame)                                     # reference 5.6 Å, frame 4.5 Å
        # the neighbor grid is enlarged to cover the distorted pairs (reference distance up to 2 dmax farther)
        @test strc.point_grid.grid_size ≈ Hamster.rcut_search(conf_frame, 1.1)
        c, basis, _ = dimer_model(strc, conf_frame, dir, "frame")
        h = Hamster.get_geometry_tensor(strc, basis, c)
        R0 = Hamster.findR0(strc.Rs)
        @test nnz_total(h) > 2
        @test any(!iszero(h[v, R0][1, 2]) for v in axes(h, 1))                # hopping of the 4.5 Å pair
        # the sparse iterator of the Hamiltonian uses the same pair set
        it = Hamster.get_sparse_iterator(strc, basis, c)
        @test any(==((1, 2)), it[R0])
        # bonds agree
        bonds = Hamster.get_bonds(strc, basis, c)
        @test any(!iszero, bonds[R0][1][1, 2])
        # the ML descriptor uses the same pair set
        set_value!(c, "model", "ML", true)
        set_value!(c, "apply_distortion", "ML", true)
        set_value!(c, "apply_distance_distortion", "ML", true)
        model = TBModel(strc, basis, c)
        d_frame = Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, c)
        @test nnz(d_frame[R0]) > 0
        conf_ref = dimer_conf(selection="reference")
        strc_ref = dimer_structure(conf_ref)
        c_ref, basis_ref, model_ref = dimer_model(strc_ref, conf_ref, dir, "ref2")
        set_value!(c_ref, "model", "ML", true)
        set_value!(c_ref, "apply_distortion", "ML", true)
        set_value!(c_ref, "apply_distance_distortion", "ML", true)
        model_ref = TBModel(strc_ref, basis_ref, c_ref)
        d_ref = Hamster.get_tb_descriptor(model_ref.hs, model_ref.params, strc_ref, basis_ref, c_ref)
        @test nnz(d_frame[R0]) > nnz(d_ref[R0])
        # label statistics see the pair as well
        stats = Hamster.nn_label_statistics(strc, basis, c)
        @test stats["Si+Si"][2] + stats["Si+Si"][3] > 0
    end
end

@testset "frame requires ML distortion flags" begin
    mktempdir() do dir
        conf = dimer_conf(selection="frame")
        set_value!(conf, "model", "ML", true)                                  # ML block present, flags default to false
        strc = dimer_structure(conf)                                           # structure construction is fine now
        c, basis, model = dimer_model(strc, conf, dir, "ml")
        @test_throws ErrorException Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, c)
        set_value!(c, "apply_distance_distortion", "ML", true)
        @test_throws ErrorException Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, c)
        set_value!(c, "apply_distortion", "ML", true)
        Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, c)     # both true: no error
        # pure TB run (no ML block): no error
        c_tb, basis_tb, model_tb = dimer_model(strc, dimer_conf(selection="frame"), dir, "tb")
        Hamster.get_tb_descriptor(model_tb.hs, model_tb.params, strc, basis_tb, c_tb)
    end
end

@testset "rcut_pair_selection = frame: H independent of the reference" begin
    mktempdir() do dir
        # Same distorted geometry (4.5 Å), two different references: 5.4 Å (within rcut) and 6.5 Å (beyond rcut)
        conf_frame = dimer_conf(selection="frame")
        conf_ref = dimer_conf(selection="reference")
        hs_frame, hs_ref = [], []
        for r_ref in (5.4, 6.5)
            for (conf, sel) in ((conf_frame, "frame"), (conf_ref, "reference"))
                strc = dimer_structure(conf, r_ref=r_ref, r_dist=4.5)
                c, basis, _ = dimer_model(strc, conf, dir, "$(r_ref)_$sel")
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

@testset "frame: H and ML descriptors independent of the reference with sepNN and ML" begin
    mktempdir() do dir
        mk(selection, r_ref) = begin
            conf = dimer_conf(selection=selection, sepNN=true)
            set_value!(conf, "model", "ML", true)
            set_value!(conf, "apply_distortion", "ML", true)
            set_value!(conf, "apply_distance_distortion", "ML", true)
            strc = dimer_structure(conf, r_ref=r_ref, r_dist=4.5)
            c, basis, model = dimer_model(strc, conf, dir, "$(selection)_$(r_ref)")
            (strc=strc, c=c, basis=basis, h=Hamster.get_geometry_tensor(strc, basis, c),
             d=Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, c),
             env=Hamster.get_environmental_descriptor(model.hs, model.params, strc, basis, c))
        end
        a54, a65 = mk("frame", 5.4), mk("frame", 6.5)
        # frame mode: same geometry tensor, descriptor and environmental descriptor for both references
        @test all(Matrix(x) == Matrix(y) for (x, y) in zip(a54.h, a65.h))
        @test all(Matrix(x) == Matrix(y) for (x, y) in zip(a54.d, a65.d))
        @test a54.env == a65.env
        # reference mode: the sepNN shell cutoffs and H still depend on the reference
        b54, b65 = mk("reference", 5.4), mk("reference", 6.5)
        @test nnz_total(b54.h) != nnz_total(b65.h) || !all(Matrix(x) == Matrix(y) for (x, y) in zip(b54.h, b65.h))
    end
end
