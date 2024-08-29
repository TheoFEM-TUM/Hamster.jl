using Hamster, LinearAlgebra, BenchmarkTools, SparseArrays

N = 32
Hr1 = rand(N, N, 25)
Hr2 = [Hr1[:, :, R] for R in axes(Hr1, 3)]
Hr2 = [sprand(N, N, 0.001) for R in axes(Hr1, 3)]

Rs = rand(3, 25); ks = rand(3, 80)

@time Hamster.get_hamiltonian(Hr1, Rs, ks)


@time Hk = Hamster.get_hamiltonian(Hr2, Rs, ks, sp_mode=true)

@time diagonalize(Hk, Neig=6)



