using Hamster, LinearAlgebra, BenchmarkTools, SparseArrays, OhMyThreads, Distributed, CubicSplines
using StaticArrays

x1 = SVector{8}(rand(8))

x2 = sparse(hcat([[SVector{8}(rand(8)) for i in 1:8] for j in 1:8]...))

Hamster.exp_sim.(x1, x2)

conf = get_config()

Hamster.get_sc_poscar(conf)

set_value!(conf, "poscar", "Supercell", "POSCAR")
