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

using Plots

nodes = [1, 2, 3, 4, 5, 6, 7, 8]
times = [1169.55, 1125.57, 1134.08, 1154.08, 1140.16, 1153.78, 1145.19, 1129.07]
plot(nodes, times ./ maximum(times), xlabel="Nodes", ylabel="Efficiency", label="Hamster", marker=:circle, ylim=(0, 1.5), labelfontsize=18, tickfontsize=18, legendfontsize=16, framestyle=:box)
hline!([1], label="Ideal scaling", linestyle=:dash, color=:red)
savefig("hamster_weak_scaling.pdf")

nodes = [1, 2, 3, 4, 5, 6, 7, 8]
times = [3026.73, 1525.60, 1030.58, 815.85, 674.28, 582.76, 521.05, 449.01] # total
#times = [1134.74, 571.15, 366.50, 300.11, 238.41, 202.53, 179.31, 152.35] #forward
#times = [1886.17, 947.41, 656.37, 509.17, 427.96, 373.14, 333.36, 287.80] # backward
#times = [5.82, 7.04, 7.70, 6.57, 7.92, 7.09, 8.38, 8.87] # update
plot(nodes, times[1] ./ times, xlabel="Nodes", ylabel="Speedup", marker=:circle, labelfontsize=18, tickfontsize=18, legendfontsize=16, framestyle=:box, xlim=(0.9,8.1), ylim=(0.9,8.1), label="Hamster")
plot!(nodes, nodes, label="Ideal scaling", linestyle=:dash, color=:red)
savefig("hamster_strong_scaling.pdf")