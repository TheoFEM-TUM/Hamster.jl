using Hamster, LinearAlgebra, BenchmarkTools, SparseArrays, OhMyThreads, Distributed, CubicSplines
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

function func4(Hks)
    Es = zeros(ComplexF64, size(Hks[1], 1), length(Hks))
    Threads.@threads for k in eachindex(Hks) 
        @views Es[:, k] = eigvals(Hks[k])
    end
end

Hks = [rand(512, 512) for k in 1:64]

addprocs(1)
@show nworkers()
begin
    @btime func1($Hks)
    sleep(5)
    GC.gc()
    @btime func2($Hks)
    sleep(5)
    GC.gc()
    @btime func3($Hks)
    sleep(5)
    GC.gc()
    @btime func4($Hks)    
end

rmprocs(workers())

BLAS.get_num_threads()
BLAS.set_num_threads(8)
Threads.nthreads()

mapslices
StructArray

H_sp = sprand(64, 64, 0.01)
H = rand(64, 64)

function hellman_feynman_step(Ψ_mk::AbstractVector, dHk_dHr)
    dE_dHr = zeros(length(Ψ_mk), length(Ψ_mk))
    for i in eachindex(Ψ_mk), j in eachindex(Ψ_mk)
        dE_dHr[i, j] = real(conj(Ψ_mk[i]) * dHk_dHr * Ψ_mk[j])
    end
    return dE_dHr
end

vs = rand(ComplexF64, 1024)
dHk_dHr = rand()

@btime hellman_feynman_step($vs, $dHk_dHr)


for i in 1:10
    i
end |> y

result = (x for x in 1:10) |> Tuple

@time Hamster.get_rcut(conf)

function func(x::Array{Float64, N}) where {N<3}
    @show x
end

using LinearAlgebra, BenchmarkTools, OhMyThreads, Distributed

@show BLAS.get_num_threads()

M = [rand(128, 128) for i in 1:8]

addprocs(4)
@btime pmap(eigvals, M)
rmprocs(workers)


@btime map(eigvals, M)
@btime tmap(eigvals, M)

function myfunc(f, N)
    M = [rand(N, N) for i in 1:48]
    f(eigvals, M)
end

@btime myfunc($tmap, 128)

38.8 / 0.86
