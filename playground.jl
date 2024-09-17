using Hamster, LinearAlgebra, BenchmarkTools, SparseArrays, OhMyThreads, Distributed, MethodAnalysis

using PrecompileTools

PrecompileTools.verbose[] = true   # runs the block even if you're not precompiling, and print precompiled calls

include("src/Hamster.jl")

Hks = [rand(ComplexF64, 8, 8) for k in 1:80]
@time diagonalize(Hks)

Hr = [rand(1024, 1024) for R in 1:25]
ks = rand(3, 80)
Rs = rand(3, 25)

@time Hamster.get_empty_hamiltonians(8, 25, Val{:dense})
@code_warntype Hamster.get_empty_hamiltonians(8, 25, Val{:sparse}, ComplexF64)

@time get_hamiltonian(Hr, Rs, ks)
@code_warntype get_hamiltonian(Hr, Rs, ks)

@code_typed get_hamiltonian(Hr, Rs, ks)

methodinstance(get_hamiltonian, (typeof(Hr), typeof(Rs), typeof(ks)))
println.(methodinstances(get_hamiltonian))

@time Hr[1] + Hr[2]

f(H1, H2) =  sum(Hr)

@btime f($Hr[1], $Hr[2])

for v in 1:NV, R in 1:NR
    @. Hr[R] += h[v, R] * V[v]
end

using PeriodicTable

elements[1].symbol

@show sort([SVector{3}([1., -1., -1.]), SVector{3}([-1., 1., -1.]), SVector{3}([-1., -1., 1.]), SVector{3}([1., 1., 1.])], rev=true)

v = [1, 2, 3]; w = [1, 4, 5]
@btime Hamster.normdiff($v, $w)


normdiff2(x...) = mapreduce(-, +, x...)

@code_warntype normdiff2(v, w)
@btime normdiff2($v, $w)


function test(vs...)
    out = zero(promote_type(eltype.(vs)...))
    for v in zip(vs...)
        out += reduce(-, v)^2
    end
    return √out
end

@code_warntype test(v, w)
@btime test($v, $w)

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