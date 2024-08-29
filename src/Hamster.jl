module Hamster

using TensorOperations, LinearAlgebra, SparseArrays, KrylovKit

include("parse/utils.jl")
include("conf/config.jl"); include("conf/read_config.jl")

include("model/hamiltonian.jl")

export Config, get_config

export get_hamiltonian, diagonalize

end # module Hamster
