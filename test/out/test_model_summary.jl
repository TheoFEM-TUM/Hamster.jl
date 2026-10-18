function get_summary_test_config()
    path = joinpath(@__DIR__, "..", "model", "test_files")
    conf = get_empty_config()
    set_value!(conf, "orbitals", "Cs", "s")
    set_value!(conf, "orbitals", "Pb", "s px py pz")
    set_value!(conf, "orbitals", "Br", "px py pz")
    set_value!(conf, "qeff", "Cs", 1)
    set_value!(conf, "qeff", "Pb", 2)
    set_value!(conf, "qeff", "Br", -1)
    set_value!(conf, "poscar", joinpath(path, "POSCAR_cspbbr3_distorted"))
    set_value!(conf, "load_rllm", true)
    set_value!(conf, "rllm_file", joinpath(path, "rllm_cspbbr3.dat"))
    set_value!(conf, "rcut", 5.5)
    set_value!(conf, "verbosity", 0)
    set_value!(conf, "ewald", true)
    set_value!(conf, "npoints", "ML", 40)
    set_value!(conf, "ncluster", "ML", 10)
    set_value!(conf, "init_params", "ML", "rand")
    return conf
end

summary_block(out, title) = (i = findfirst(title, out); isnothing(i) ? "" : out[i[1]:findnext("\n" * "="^80, out, i[1])[1]])

@testset "Model summary helpers" begin
    xs = [SVector{8}(zeros(8)), SVector{8}(fill(0.1, 8))]
    ds = [SVector{8}(zeros(8))]
    @test Hamster.max_similarities(xs, ds, 0.1) ≈ [1.0, exp(-0.08 / 0.02)]
    @test Hamster.max_similarities(xs, typeof(ds)(), 0.1) == [0.0, 0.0]
    counts = zeros(Int64, 1000); counts[100] = 1; counts[900] = 3
    @test Hamster.histogram_quantile(counts, 0.25) == 0.1
    @test Hamster.histogram_quantile(counts, 0.5) == 0.9
    @test Hamster.range_string(3, 3) == "3" && Hamster.range_string(3, 5) == "3 - 5"
end

@testset "Model summaries in hamster.out" begin
    conf = get_summary_test_config()
    strc = Structure(conf)
    basis = Basis(strc, conf)

    mktempdir() do dir
        cd(dir) do
            touch("hamster.out")
            ham = EffectiveHamiltonian([strc], [basis], comm, conf, label="training set")
            ham_val = EffectiveHamiltonian([strc], [basis], comm, conf, label="validation set",
                                           ml_data_points=Hamster.get_ml_data_points(ham, conf))
            EffectiveHamiltonian([strc], [basis], comm, conf, label="skipped", write_summary=false)
            out = read("hamster.out", String)
            @test !occursin("skipped", out)

            tb = summary_block(out, "TB Model Summary (training set)")
            tb_model = ham.models[1]
            @test occursin("atoms per structure:         $(length(strc.ions))", tb)
            @test occursin("orbitals per structure:      $(length(basis)) (total $(length(basis)))", tb)
            @test occursin("parameters:                  $(length(tb_model.param_labels))", tb)
            @test occursin("onsite $(count(p -> p.nnlabel == 0, tb_model.param_labels))", tb)
            @test occursin("Br+Pb", tb)

            ml = summary_block(out, "ML Model Summary (training set)")
            ndescriptors = sum(nnz, ham.models[2].structure_descriptors[1])
            @test occursin("descriptor vectors:          $ndescriptors", ml)
            for group in ["Cs onsite", "Pb onsite", "Br onsite", "Br+Pb"]
                @test occursin(Regex("\\n    " * replace(group, "+" => "\\+") * "\\s+\\d+"), ml)
            end
            @test occursin("ML Model Summary (validation set)", out)
            @test ham_val.models[2].data_points == ham.models[2].data_points

            ewald = summary_block(out, "Ewald Model Summary (training set)")
            @test occursin("structures:                  1", ewald)
            @test !occursin("qeff", ewald) && !occursin("rebuilt", ewald)
            for element in ["Cs", "Pb", "Br"]
                @test occursin(Regex("\\n    $element\\s+\\d+ "), ewald)
            end
        end
    end
end

@testset "Ewald neighbor grid for large rcut" begin
    conf = get_summary_test_config()
    strc = Structure(conf)
    basis = Basis(strc, conf)
    mktempdir() do dir
        cd(dir) do
            touch("hamster.out")
            model = EwaldOnsites([strc], [basis], comm, conf, rcut=9.0, label="large rcut")
            @test occursin("real-space neighbor grid:    rebuilt", read("hamster.out", String))

            pos = Hamster.get_ion_positions(strc.ions, apply_distortion=true)
            q = [Hamster.get_qeff(conf, Hamster.number_to_element(t)) for t in Hamster.get_ion_types(strc.ions)]
            φ = Hamster.ewald_sum(pos, q, strc.lattice, rcut=9.0).potentials
            @test model.potentials[1] ≈ (φ .- [mean(φ[Hamster.get_ion_types(strc.ions) .== t]) for t in Hamster.get_ion_types(strc.ions)])
        end
    end
end

@testset "Validation kernel with training data points" begin
    conf = get_summary_test_config()
    strc = Structure(conf)
    basis = Basis(strc, conf)
    tb_model = TBModel([strc], [basis], comm, conf)

    kernel = HamiltonianKernel([strc], [basis], tb_model, comm, conf, write_summary=false)
    kernel_pre = Hamster.HamiltonianKernelPrecalced(kernel, 0, [""], conf)
    val = HamiltonianKernel([strc], [basis], tb_model, comm, conf, data_points=kernel.data_points, write_summary=false)
    val_pre = HamiltonianKernel([strc], [basis], tb_model, comm, conf, data_points=kernel_pre.kp.datapoints, sim_mat=true,
                                write_summary=false)
    Hamster.copy_params!(val, kernel)
    Hamster.copy_params!(val_pre, kernel_pre)
    Δ(A, B) = maximum(maximum(abs.(A[R] .- B[R])) for R in eachindex(A))
    @test val.data_points == kernel.data_points
    @test Δ(get_hr(val, Hamster.Dense(), 1), get_hr(kernel, Hamster.Dense(), 1)) < 1e-12
    @test Δ(get_hr(val_pre, Hamster.Dense(), 1), get_hr(kernel_pre, Hamster.Dense(), 1)) < 1e-12
end
