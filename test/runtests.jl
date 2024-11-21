using Hamster, Test, LinearAlgebra, SparseArrays, StaticArrays, HCubature, Statistics, FiniteDiff, HDF5, TensorOperations, Suppressor

@testset "Parse" begin
    include("parse/test_utils.jl")
    include("parse/test_poscar.jl")
end

@testset "Config" begin
    include("conf/test_config.jl")
    include("conf/test_read_config.jl")
end

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

@testset "model" begin
    include("model/test_ham.jl")
    include("model/test_ham_grad.jl")
    include("model/test_model.jl")
    include("model/test_eff_ham.jl")
end

@testset "optim" begin
    include("optim/test_loss.jl")
    include("optim/test_data.jl")
    include("optim/test_profiler.jl")
    include("optim/test_optimize.jl")
end