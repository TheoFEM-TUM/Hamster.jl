using Hamster, Test, LinearAlgebra, SparseArrays, StaticArrays

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
end

@testset "model" begin
    include("model/test_ham.jl")
end
