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

function func1(dL_dE, dE_dHr, mode)
    dL_dHr = Hamster.get_empty_real_hamiltonians(size(dE_dHr, 2), size(dE_dHr, 1), mode)
    tforeach(axes(dE_dHr, 3), nchunks=16) do k
        for m in axes(dE_dHr, 2)
            for R in axes(dE_dHr, 1)
                @views @. dL_dHr[R] += dL_dE[m , k] * dE_dHr[R, m, k]
            end
        end
    end
    return dL_dHr
end

function func2(dL_dE, dE_dHr, mode; kpar=Threads.nthreads(), mpar=Threads.nthreads())
    dL_dHr = Hamster.get_empty_real_hamiltonians(size(dE_dHr, 2), size(dE_dHr, 1), mode)
    tforeach(axes(dE_dHr, 3), nchunks=kpar) do k
        tforeach(axes(dE_dHr, 2), nchunks=mpar) do m
            for R in axes(dE_dHr, 1)
                @views @. dL_dHr[R] += dL_dE[m , k] * dE_dHr[R, m, k]
            end
        end
    end
    return dL_dHr
end

function func3(dL_dE, dE_dHr, mode)
    dL_dHr = Hamster.get_empty_real_hamiltonians(size(dE_dHr, 2), size(dE_dHr, 1), mode)
    @tasks for k in axes(dE_dHr, 3)
        @tasks for m in axes(dE_dHr, 2)
            for R in axes(dE_dHr, 1)
                @views @. dL_dHr[R] += dL_dE[m , k] * dE_dHr[R, m, k]
            end
        end
    end
    return dL_dHr
end

Nε = 128; Nk = 1; NR = 25

dL_dE = zeros(Nε, Nk)
dE_dHr = Array{Matrix{Float64}, 3}(undef, NR, Nε, Nk)
for k in 1:Nk, m in 1:Nε, R in 1:NR
    dE_dHr[R, m, k] = rand(Nε, Nε)
end

BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
GC.gc()
@btime func1($dL_dE, $dE_dHr, $Hamster.Dense())
@btime func2($dL_dE, $dE_dHr, $Hamster.Dense(), kpar=1, mpar=16)
@btime func3($dL_dE, $dE_dHr, $Hamster.Dense())


@btime func3($dL_dE3, $dE_dHr3, $Hamster.Dense())

dL_dE3 = zeros(Nk, Nε)
dE_dHr3 = Array{Matrix{Float64}, 3}(undef, NR, Nk, Nε)
for k in 1:Nk, m in 1:Nε, R in 1:NR
    dE_dHr3[R, k, m] = rand(Nε, Nε)
end

function func3(dL_dE, dE_dHr, mode)
    dL_dHr = Hamster.get_empty_real_hamiltonians(size(dE_dHr, 3), size(dE_dHr, 1), mode)
    @tasks for m in axes(dE_dHr, 3)
        for k in axes(dE_dHr, 2)
            for R in axes(dE_dHr, 1)
                @views @. dL_dHr[R] += dL_dE[k, m] * dE_dHr[R, k, m]
            end
        end
    end
    return dL_dHr
end

map(enumerate(5:10)) do (i, ind)
    println(i," ", ind)
end