using Hamster, Test, LinearAlgebra, SparseArrays, StaticArrays, HCubature, Statistics, 
FiniteDiff, HDF5, TensorOperations, Suppressor, MPI, Logging, BlockDiagonals

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

args = Hamster.parse_commandline(ARGS)
test_all = !haskey(args, "testset")
test_only = test_all ? "all" : args["testset"]

if test_all || test_only == "Parse"
    @testset "Parse" begin
        include("parse/test_utils.jl")
        include("parse/test_poscar.jl")
        include("parse/test_wannier90.jl")
        include("parse/test_commandline.jl")
    end
end

if test_all || test_only == "Config"
    @testset "Config" begin
        include("conf/test_config.jl")
        include("conf/test_read_config.jl")
        include("conf/test_defaults.jl")
    end
end

if test_all || test_only == "Structure"
    @testset "Structure" begin
        include("strc/test_vec.jl")
        include("strc/test_grid.jl")
        include("strc/test_lattice.jl")
        include("strc/test_ion.jl")
        include("strc/test_sk_transform.jl")
        include("strc//test_structure.jl")
        include("strc/test_methods.jl")
        include("strc/test_supercell.jl")
    end
end

if test_all || test_only == "Basis"
    @testset "Basis" begin
        include("basis/test_spher_harm.jl")
        include("basis/test_orbconfig.jl")
        include("basis/test_adaptive_intp.jl")
        include("basis/test_orbital.jl")
        include("basis/test_label.jl")
        include("basis/test_index.jl")
        include("basis/test_overlap.jl")
        include("basis/test_param.jl")
        include("basis/test_rllm.jl")
        include("basis/test_basis.jl")
    end
end

if test_all || test_only == "Model"
    @testset "Model" begin
        include("model/test_ham.jl")
        include("model/test_ham_grad.jl")
        include("model/test_model.jl")
        include("model/test_eff_ham.jl")
        include("model/test_ham_write.jl")
    end
end

if test_all || test_only == "Optim"
    @testset "Optim" begin
        include("optim/test_loss.jl")
        include("optim/test_data.jl")
        include("optim/test_profiler.jl")
        include("optim/test_optimize.jl")
    end
end

if test_all || test_only == "ML"
    @testset "ML" begin
        include("mlham/test_descriptor.jl")
        include("mlham/test_kernel.jl")
    end
end

if test_all || test_only == "SOC"
    @testset "SOC" begin
        include("soc/test_soc_utils.jl")
        include("soc/test_soc_matrices.jl")
        include("soc/test_soc_model.jl")
    end
end

if test_all || test_only == "calc"
    @testset "calc" begin
        include("calc/test_optimization.jl")
        include("calc/test_standard.jl")
        include("calc/test_hyperopt.jl")
    end
end

MPI.Finalize()