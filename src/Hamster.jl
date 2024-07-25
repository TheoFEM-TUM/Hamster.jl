module Hamster

using TensorOperations, LinearAlgebra, SparseArrays

include("parse/utils.jl")
include("conf/config.jl"); include("conf/read_config.jl")

include("model/hamiltonian.jl")

export Config, get_config

end # module Hamster
