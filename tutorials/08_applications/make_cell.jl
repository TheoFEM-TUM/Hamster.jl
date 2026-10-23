# Builds a large GaAs cell for tutorial 8: n × n × n conventional cells (8 atoms each) with random displacements of the
# atoms (Gaussian, standard deviation σ in Å), a crude stand-in for a snapshot of a large MD run.
# Usage: julia make_cell.jl [n] [σ]   (default: 4 → 512 atoms, σ = 0.1 Å); writes POSCAR_large
using Random, Printf
n = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 4
σ = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 0.1
a = 5.65                                                    # lattice constant of GaAs (Å), as in data/gaas/SC_POSCAR
fcc = [0 0 0; 0.5 0.5 0; 0.5 0 0.5; 0 0.5 0.5]
basis = Dict("Ga" => fcc, "As" => fcc .+ 0.25)
Random.seed!(1)
open("POSCAR_large", "w") do io
    println(io, "GaAs $(n)x$(n)x$(n) conventional cells, random displacements σ = $σ Å")
    println(io, "  1.0")
    for i in 1:3
        println(io, join([@sprintf("%14.8f", i == j ? n * a : 0.0) for j in 1:3]))
    end
    println(io, "   Ga   As\n", "   ", 4n^3, "   ", 4n^3, "\nDirect")
    for el in ("Ga", "As"), i in 0:n-1, j in 0:n-1, k in 0:n-1, r in eachrow(basis[el])
        frac = (r .+ [i, j, k]) ./ n .+ σ .* randn(3) ./ (n * a)
        println(io, join([@sprintf("%14.8f", mod(x, 1)) for x in frac]))
    end
end
println("POSCAR_large written: ", 8n^3, " atoms")
