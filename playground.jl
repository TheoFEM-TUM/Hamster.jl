using Hamster, LinearAlgebra, BenchmarkTools, SparseArrays, OhMyThreads, Distributed, MethodAnalysis, CubicSplines
using StaticArrays

w = Float64[]

@show isempty(w)

h = rand(34, 25)

for ind in CartesianIndices(h)
    h[ind]
end

ehm = Hamster.EffectiveHamiltonian(rand(3), rand(3))

Hr = [rand(8, 8) for i in 1:25]

@show size.(Hamster.apply_spin_basis.(Hr))

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



vs = rand(8, 8, 80)
ex = rand(25, 80)

Threads.@threads for k in axes(ex, 2)
    Threads.@threads for R in axes(ex, 1)
        for m in axes(vs, 2)
            product = @. conj(vs[:, m, k])' * ex[R, k] * vs[:, m, k]
            @show size(product)
        end
    end
end

v = sprand(8, 0.1)
product = @. conj(v)' * ex[1, 1] * v


using Statistics
reduce(mean, [1, 2, 3])

using BenchmarkTools

f1(x) = cos(x)^2
f2(x) = cos(x) * cos(x)

x = rand(10000)

@btime f1.($x)
@btime f2.($x)