using LinearAlgebra

# Pair cutoffs rcut_<A>_<B>: a cutoff per pair of elements, 0 for no interaction between different ions of the pair

# Two Si atoms 3.0 Å apart and a C atom 3.0 Å from the first Si (4.24 Å from the second) in a large cubic cell
function sic_structure(conf; a=20.0)
    lattice = a * Matrix(1.0I, 3, 3)
    rs = [0.0 3.0 0.0; 0.0 0.0 3.0; 0.0 0.0 0.0]
    types = Hamster.element_to_number.(["Si", "Si", "C"])
    Rs = Hamster.get_translation_vectors(rs, lattice, rcut=Hamster.rcut_search(conf))
    return Structure(Rs, rs, zeros(3, 3), types, lattice, conf)
end

function sic_conf(; pairs=Dict{String, Float64}(), onsite=true, selection="reference")
    conf = get_empty_config()
    set_value!(conf, "rcut", 5.5)
    set_value!(conf, "rcut_pair_selection", selection)
    set_value!(conf, "onsite", onsite)
    set_value!(conf, "verbosity", 0)
    for (key, value) in pairs
        set_value!(conf, key, value)
    end
    for (element, n, alpha) in (("Si", 3, 10), ("C", 2, 14))
        set_value!(conf, "orbitals", element, "s")
        set_value!(conf, "n", element, n)
        set_value!(conf, "alpha", element, alpha)
    end
    return conf
end

function sic_model(conf, dir, name)
    c = deepcopy(conf)
    set_value!(c, "rllm_file", joinpath(dir, "rllm_$name.dat"))
    strc = sic_structure(c)
    basis = Basis(strc, c)
    Hamster.get_rllm(basis.overlaps, c)
    model = TBModel(strc, basis, c)
    H = Matrix(Hamster.get_hr(model.hs, ones(length(model.params)), Hamster.Sparse())[Hamster.findR0(strc.Rs)])
    return (strc=strc, basis=basis, conf=c, model=model, H=H)
end

param_value(model, name) = model.params[findfirst(==(name), string.(model.param_labels))]

@testset "pair cutoff tags" begin
    @test Hamster.pair_rcut_elements("rcut_Pb_Br") == ("Br", "Pb")
    @test Hamster.pair_rcut_elements("rcut_br_pb") == ("Br", "Pb")
    @test Hamster.pair_rcut_elements("rcut_Pb_Pb") == ("Pb", "Pb")
    @test isnothing(Hamster.pair_rcut_elements("rcut_tol"))
    @test isnothing(Hamster.pair_rcut_elements("rcut_pair_selection"))
    @test isnothing(Hamster.pair_rcut_elements("rcut_Xx_Pb"))
    conf = sic_conf(pairs=Dict("rcut_Si_Si" => 0.0, "rcut_C_Si" => 4.0))
    @test Hamster.get_pair_rcuts(conf) == Dict(("Si", "Si") => 0.0, ("C", "Si") => 4.0)
    M = Hamster.get_rcut_matrix(conf)
    si, c = Hamster.element_to_number("Si"), Hamster.element_to_number("C")
    @test M[si, si] == 0 && M[si, c] == M[c, si] == 4.0 && M[c, c] == 5.5
    @test Hamster.get_rcut_matrix(conf; rcut=3.0)[si, c] == 3.0              # at most rcut (e.g. the ML rcut)
    # both orders of the elements: the same value is fine, different values and values beyond [0, rcut] are errors
    @test Hamster.get_pair_rcuts(sic_conf(pairs=Dict("rcut_C_Si" => 4.0, "rcut_Si_C" => 4.0))) == Dict(("C", "Si") => 4.0)
    @test_throws ErrorException Hamster.get_pair_rcuts(sic_conf(pairs=Dict("rcut_C_Si" => 4.0, "rcut_Si_C" => 3.0)))
    @test_throws ErrorException Hamster.get_pair_rcuts(sic_conf(pairs=Dict("rcut_C_Si" => 6.0)))
    @test_throws ErrorException Hamster.get_pair_rcuts(sic_conf(pairs=Dict("rcut_C_Si" => -1.0)))
    mktempdir() do dir
        file = joinpath(dir, "hconf")
        write(file, "begin Options\n    rcut = 5.5\n    rcut_Pb_Pb = 0\n    rcut_Br_Pb = 4.5\nend\n")
        @test_logs get_config(filename=file)                                    # known tags: no warnings
        write(file, "begin Options\n    rcut = 5.5\n    rcut_Br_Pb = 6\nend\n")
        @test_throws r"between 0 and rcut" get_config(filename=file)            # checked when the config is read
        @test Hamster.hyperopt_tag("rcut_Br_Pb") == ("rcut_Br_Pb", "Options")   # a hyperparameter of the Options block
    end
end

@testset "pair_in_cutoff and pair_weight with cutoff 0" begin
    @test Hamster.pair_in_cutoff(0.0, 0.0, 0.0, 1.0, "reference")           # on-site term
    @test Hamster.pair_in_cutoff(0.0, 0.0, 0.0, -1.0, "frame")
    @test !Hamster.pair_in_cutoff(3.0, 3.0, 0.0, 1.0, "reference")          # different ions
    @test !Hamster.pair_in_cutoff(3.0, 0.5, 0.0, 1.0, "frame")
    @test Hamster.pair_weight(0.0, 0.0, -1.0) == 1.0                        # no smooth cutoff for cutoff 0
    @test Hamster.pair_weight(3.0, 4.0, 1.0) == Hamster.fcut(3.0, 4.0, 1.0)
end

@testset "pair cutoffs in the Hamiltonian" begin
    mktempdir() do dir
        ref = sic_model(sic_conf(), dir, "ref")
        # pair cutoffs equal to rcut give the same geometry tensor as without them (models without the tags)
        same = sic_model(sic_conf(pairs=Dict("rcut_Si_Si" => 5.5, "rcut_C_Si" => 5.5)), dir, "same")
        @test all(Matrix(a) == Matrix(b) for (a, b) in zip(ref.model.hs, same.model.hs))
        @test !iszero(ref.H[1, 2]) && !iszero(ref.H[1, 3]) && !iszero(ref.H[2, 3])
        for selection in ("reference", "frame")
            # Si-Si cutoff 0: no Si-Si hopping, on-site terms and Si-C unchanged
            cut = sic_model(sic_conf(pairs=Dict("rcut_Si_Si" => 0.0), selection=selection), dir, "si0_$selection")
            @test iszero(cut.H[1, 2]) && iszero(cut.H[2, 1])
            @test cut.H[1, 1] == ref.H[1, 1] && cut.H[3, 3] == ref.H[3, 3]
            @test cut.H[1, 3] == ref.H[1, 3] && cut.H[2, 3] == ref.H[2, 3]
            R0 = Hamster.findR0(cut.strc.Rs)
            @test !any(==((1, 2)), Hamster.get_sparse_iterator(cut.strc, cut.basis, cut.conf)[R0])
            @test iszero(Hamster.get_bonds(cut.strc, cut.basis, cut.conf)[R0][1][1, 2])
            # Si-C cutoff 3.5 (smooth cutoff to 4.5 Å): the 3.0 Å pair keeps its value; the 4.24 Å pair is damped in
            # frame mode and beyond the cutoff of the reference (no distortion) in reference mode
            short = sic_model(sic_conf(pairs=Dict("rcut_C_Si" => 3.5), selection=selection), dir, "sic35_$selection")
            @test short.H[1, 3] == ref.H[1, 3]
            if selection == "frame"
                @test short.H[2, 3] ≈ ref.H[2, 3] * Hamster.fcut(sqrt(18), 3.5, 1.0)
            else
                @test iszero(short.H[2, 3])
            end
        end
        # the ML descriptor uses the pair cutoffs (at most the ML rcut)
        cut = sic_model(sic_conf(pairs=Dict("rcut_Si_Si" => 0.0)), dir, "si0_ml")
        set_value!(cut.conf, "model", "ML", true)
        d = Hamster.get_tb_descriptor(cut.model.hs, cut.model.params, cut.strc, cut.basis, cut.conf)
        R0 = Hamster.findR0(cut.strc.Rs)
        @test iszero(d[R0][1, 2]) && !iszero(d[R0][1, 3])
    end
end

@testset "parameters of excluded pairs are zero" begin
    mktempdir() do dir
        cut = sic_model(sic_conf(pairs=Dict("rcut_Si_Si" => 0.0)), dir, "params")
        @test param_value(cut.model, "NN1_Si+Si_ssσ") == 0
        @test param_value(cut.model, "NN0_Si+Si_diag0") == 1                   # on-site parameter kept
        @test param_value(cut.model, "NN1_C+Si_ssσ") == 1
        # onsite = false: the on-site terms of Si use the NN1 parameter, which is kept
        nos = sic_model(sic_conf(pairs=Dict("rcut_Si_Si" => 0.0), onsite=false), dir, "nos")
        @test param_value(nos.model, "NN1_Si+Si_ssσ") == 1
        @test iszero(nos.H[1, 2]) && !iszero(nos.H[1, 1])
        # also from a parameter file with nonzero values, and written as zero
        file = joinpath(dir, "params_in")
        Hamster.write_params(cut.model.param_labels, fill(0.5, length(cut.model.params)), sic_conf(); filename=file)
        set_value!(cut.conf, "init_params", file * ".dat")
        @test_logs (:warn, r"rcut_Si_Si") Hamster.init_params!(cut.model, cut.conf)   # header without the pair cutoff
        @test param_value(cut.model, "NN1_Si+Si_ssσ") == 0 && param_value(cut.model, "NN1_C+Si_ssσ") == 0.5
    end
end

@testset "pair cutoffs in the parameter file headers" begin
    mktempdir() do dir
        conf = sic_conf(pairs=Dict("rcut_Si_Si" => 0.0, "rcut_Si_C" => 4.0))
        cut = sic_model(conf, dir, "header")
        file = joinpath(dir, "params")
        Hamster.write_params(cut.model.param_labels, cut.model.params, conf; filename=file)
        text = read(file * ".dat", String)
        @test occursin("rcut_C_Si = 4.0", text) && occursin("rcut_Si_Si = 0.0", text)
        _, _, _, _, header = Hamster.read_params(file * ".dat")
        @test Hamster.check_consistency(header, conf)
        @test Hamster.check_consistency(header, sic_conf(pairs=Dict("rcut_C_Si" => 4.0, "rcut_Si_Si" => 0.0)))
        @test_logs (:warn, r"rcut_C_Si: params.dat = \"4.0\", hconf = \"5.5\"") Hamster.check_consistency(header,
            sic_conf(pairs=Dict("rcut_Si_Si" => 0.0))) == false
        # a file without pair cutoffs is consistent with pair cutoffs equal to rcut
        @test Hamster.check_consistency(Dict("rcut" => "5.5"), sic_conf(pairs=Dict("rcut_C_Si" => 5.5)))
        # ML header: a different pair cutoff (as seen with the ML rcut) is an error
        ml = joinpath(dir, "ml_params.dat")
        write(ml, "begin unknown\n  rcut_pair_selection = reference\n  rcut_C_Si = 4.0\nend\n")
        @test Hamster.check_ml_settings(sic_conf(pairs=Dict("rcut_C_Si" => 4.0)), ml) === nothing
        @test_throws r"pair C-Si" Hamster.check_ml_settings(sic_conf(), ml)
        conf3 = sic_conf(); set_value!(conf3, "rcut", "ML", 3.0)                # both cut at the ML rcut
        @test Hamster.check_ml_settings(conf3, ml) === nothing
    end
end
