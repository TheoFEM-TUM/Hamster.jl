using Hamster, Test, LinearAlgebra, SparseArrays


include("conf/test_config.jl")
include("conf/test_read_config.jl")
include("parse/test_utils.jl")

@testset "model" begin
    include("model/test_ham.jl")
end
