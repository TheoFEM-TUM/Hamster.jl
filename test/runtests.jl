using Hamster, Test, LinearAlgebra, SparseArrays, StaticArrays, HCubature, Statistics

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
end

@testset "model" begin
    include("model/test_ham.jl")
end
