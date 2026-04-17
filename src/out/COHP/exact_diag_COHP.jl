using LinearAlgebra, SparseArrays
using MPI
using DelimitedFiles
include("../helper.jl")


function compute_cohp(H::Matrix{ComplexF64}, basis_labels::Vector{String}; n_E::Int = 1000, sigma::Float64 = 0.05, broadening::Symbol = :gaussian)

    EVs, vecs = eigen(Hermitian(H))  
    println("- Hamiltonian diagonalized")

    n_orb = length(EVs)
    E_grid = range(minimum(EVs) * 1.05, maximum(EVs) * 1.05, length=n_E)

    COHPs = Dict{Tuple{String,String}, Vector{Float64}}()


    for k in 1:n_orb
        dE = E_grid .- EVs[k]
        weight = broadening_weight.(dE, sigma, broadening)

        for i in 1:n_orb
            for j in i+1:n_orb
                label_i = basis_labels[i]
                label_j = basis_labels[j]

                if label_i != label_j
                    
                    H_ij = H[i, j]
                    H_ji = H[j, i]

                    if abs(H_ij) > 1e-10 || abs(H_ji) > 1e-10

                        pair_key = (label_i, label_j)

                        if !haskey(COHPs, pair_key)
                            if !haskey(COHPs, (label_j, label_i))
                                COHPs[pair_key] = zeros(Float64, n_E)
                            else
                                pair_key = (label_j, label_i)
                            end
                        end

                        ci   = vecs[i, k]
                        cj   = vecs[j, k]

                        cohp_k = real(H_ij * conj(ci) * cj + H_ji * conj(cj) * ci)

                        COHPs[pair_key] .+= cohp_k * weight

                    end

                end
            end
        end
    end


    return EVs, vecs, E_grid, COHPs

end





### ### ### ### ### ### ###
### main script execution
### ### ### ### ### ### ###

H_path            = ARGS[1]
dir_outpath       = ARGS[2]
t                 = parse(Int, ARGS[3])
hamiltonian_style = ARGS[4]

basis_labels = get_basis_labels(H_path)
#basis_labels = get_basis_labels1(H_path)
println("- basis labels: ", basis_labels)

MPI.Init()

H = get_dense_H(H_path, t, hamiltonian_style)
println("- Hamiltonian read")

EVs, vecs, E_grid, COHPs = compute_cohp(H, basis_labels)


for (pair, cohp) in COHPs
    i, j = pair

    dir_outpath_ij = joinpath(dir_outpath, "$(i)_$(j)/")
    #println(dir_outpath_ij)
    mkpath(dir_outpath_ij)
    data_file = joinpath(dir_outpath_ij, "COHP_$t.txt")

    open(data_file, "w") do io
        println(io, "# Energy   COHP")
        writedlm(io, hcat(E_grid, cohp))
    end
end

println("COHP calculation for snapshot $t completed.")

MPI.Finalize()
