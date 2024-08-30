module Hamster

using TensorOperations, LinearAlgebra, SparseArrays, KrylovKit

include("parse/utils.jl")
include("conf/config.jl"); include("conf/read_config.jl")

include("model/ham.jl"); include("model/ham_write.jl")

export write_to_file, read_from_file

export Config, get_config

export get_hamiltonian, diagonalize
export write_hr, read_hr

end # module Hamster
