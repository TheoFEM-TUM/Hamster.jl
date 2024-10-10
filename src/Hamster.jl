module Hamster

using TensorOperations, LinearAlgebra, SparseArrays, StaticArrays, KrylovKit, Dates, PeriodicTable, 
    CubicSplines, HCubature, Statistics, ChunkSplitters, Distributed

include("parse/utils.jl"); include("parse/poscar.jl")

include("conf/config.jl"); include("conf/read_config.jl")
include("conf/defaults.jl"); include("conf/strc_defaults.jl"); include("conf/basis_defaults.jl"); include("conf/optim_defaults.jl")

include("out/output.jl")

include("strc/vec.jl"); include("strc/grid.jl"); include("strc/lattice.jl"); include("strc/ion.jl"); include("strc/sk_transform.jl")
include("strc/structure.jl"); include("strc/methods.jl")

include("basis/index.jl"); include("basis/sper_harm.jl"); include("basis/sh_transforms.jl"); include("basis/orbconfig.jl")
include("basis/adaptive_intp.jl")
include("basis/label.jl"); include("basis/orbital.jl"); include("basis/overlap.jl"); include("basis/param.jl"); include("basis/rllm.jl")
include("basis/basis.jl")

include("model/ham.jl"); include("model/ham_diff.jl"); include("model/ham_write.jl"); include("model/model.jl")
include("model/eff_ham.jl")

include("optim/adam.jl"); include("optim/loss.jl")

export write_to_file, read_from_file

export Config, get_config, get_empty_config, set_value!

export findR0, Ion, Structure, get_nearest_neighbors

export ParameterLabel, read_params, write_params, Basis, get_geometry_tensor

export get_hamiltonian, diagonalize, get_hr, TBModel, init_params!
export write_hr, read_hr

export update!

# Precompililation
include("Hamster_precompile.jl")

end # module Hamster
