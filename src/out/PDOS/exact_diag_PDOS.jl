using LinearAlgebra, SparseArrays
using MPI
using DelimitedFiles
include("../helper.jl")


function compute_pdos(H::Matrix{ComplexF64}, basis_labels::Vector{String};
                      n_E::Int = 1000, sigma::Float64 = 0.05, broadening::Symbol = :gaussian)

    EVs, vecs = eigen(Hermitian(H))
    println("- Hamiltonian diagonalized")

    n_orb = length(EVs)
    E_grid = range(minimum(EVs) * 1.05, maximum(EVs) * 1.05, length=n_E)

    proj = abs2.(vecs)

    unique_labels = unique(basis_labels)
    PDOS = Dict{String, Vector{Float64}}(label => zeros(Float64, n_E) for label in unique_labels)

    E_vec = collect(E_grid)

    for k in 1:n_orb
        dE = E_vec .- EVs[k]
        weight = broadening_weight.(dE, sigma, broadening)  

        for i in 1:n_orb
            PDOS[basis_labels[i]] .+= proj[i, k] .* weight
        end
    end


    return EVs, vecs, E_grid, PDOS

end





### ### ### ### ### ### ###
### main script execution
### ### ### ### ### ### ###

H_path            = ARGS[1]
dir_outpath       = ARGS[2]
t                 = parse(Int, ARGS[3])
hamiltonian_style = ARGS[4]

#basis_labels = get_basis_labels(H_path)
basis_labels = get_basis_labels1(H_path)

println("- basis labels: ", basis_labels)

MPI.Init()

H = get_dense_H(H_path, t, hamiltonian_style)
println("- Hamiltonian read")

EVs, vecs, E_grid, PDOS = compute_pdos(H, basis_labels)


for (label, pdos) in PDOS
    dir_outpath_l = joinpath(dir_outpath, "$(label)/")
    mkpath(dir_outpath_l)
    data_file = joinpath(dir_outpath_l, "PDOS_$t.txt")
    open(data_file, "w") do io
        println(io, "# Energy   PDOS")
        writedlm(io, hcat(E_grid, pdos))
    end
end

println("PDOS calculation for snapshot $t completed.")
MPI.Finalize()