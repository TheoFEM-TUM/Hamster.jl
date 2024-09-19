using Hamster, LinearAlgebra, BenchmarkTools, SparseArrays, OhMyThreads, Distributed, MethodAnalysis


for v in 1:NV, R in 1:NR
    @. Hr[R] += h[v, R] * V[v]
end

func1(Hks) = map(eigvals, Hks)

func2(Hks) = pmap(eigvals, Hks)

func3(Hks) = tmap(eigvals, Hks)

Hks = [rand(64, 64) for k in 1:80]

addprocs(16)
@show nworkers()

@btime func1($Hks)
@btime func2($Hks)
@btime func3($Hks)

rmprocs(workers())

mapslices
StructArray

H_sp = sprand(64, 64, 0.01)
H = rand(64, 64)

@show length(eachindex(H_sp))
@show length(eachindex(H))