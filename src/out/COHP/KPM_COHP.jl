using SparseArrays, LinearAlgebra, KrylovKit
using DelimitedFiles
using Base.Threads: nthreads
using Random, Statistics, Distributions
using OhMyThreads, MPIPreferences
using MPI
include("../KPM.jl")
include("../helper.jl")



### calculate coeff_COHP_m = v * P_i * H * P_j * T_m(H) * v
function kernel_polynomial_method_cohp(H::SparseMatrixCSC{ComplexF64}, v::Vector{ComplexF64}, M::Int, basis_labels::Vector{String}, unique_labels::Vector{String})

    n_labels = length(unique_labels)

    Ps = get_projectors(basis_labels, unique_labels)

    coeff_COHP = Array{Float64}(undef, M, n_labels, n_labels)

    pi_H_pj = Array{ComplexF64}(undef, length(v), n_labels, n_labels)        

    for i in 1:n_labels
        for j in 1:n_labels
            pi_H_pj[:, i, j] = projector(H * projector(v, Ps[:, i]), Ps[:, j])
        end
    end

    v1 = copy(v) 
    v2 = H * v
    v3 = zeros(ComplexF64, length(v))

    for i in 1:n_labels
        for j in 1:n_labels
            coeff_COHP[1, i, j] = real(pi_H_pj[:, i, j] ⋅ v1)
            coeff_COHP[2, i, j] = real(pi_H_pj[:, i, j] ⋅ v2)
        end
    end

    for m in 3:M

        v3 .= 2 .* H * v2 .- v1

        for i in 1:n_labels
            for j in 1:n_labels
                coeff_COHP[m, i, j] = real(pi_H_pj[:, i, j] ⋅ v3)
            end
        end

        v1 .= v2
        v2 .= v3

    end

    return coeff_COHP
end

### compute COHPs from coefficients coeff_COHP_m
function compute_cohp(M, mean_E, ΔE, coeff_COHP, unique_labels)

    n_labels = length(unique_labels)

    g_m = [jackson_kernel_elem(m, M) for m in 1:M]

    E_grid, delta_m_E = get_delta_m_E(M, mean_E, ΔE, g_m)

    COHPs = Dict{Tuple{String,String}, Vector{Float64}}()

    for i in 1:n_labels
        for j in 1:n_labels
            elem_i = unique_labels[i]
            elem_j = unique_labels[j]

            if elem_i != elem_j
                pair_key = (elem_i, elem_j)

                if !haskey(COHPs, pair_key)
                    if !haskey(COHPs, (elem_j, elem_i))
                        COHPs[pair_key] = zeros(Float64, 2 * M)
                    else
                        pair_key = (elem_j, elem_i)
                    end
                end

                for m in 1:M
                    COHPs[pair_key] += coeff_COHP[m, i, j] * delta_m_E[:, m]
                end
            end

        end
    end

    return E_grid, COHPs
end

### main function to run KPM-COHP calculation
function KPM_COHP(H_path::String, output_dir::String, t::Int, M::Int, N::Int)

    MPI.Init()
    basis_labels = get_basis_labels(H_path)
    unique_labels = unique(basis_labels)

    ### read and rescale hamiltonian
    H = get_sparse_H(H_path, t, hamiltonian_style)
    println("- Hamiltonian read")
    E_max, E_min = get_spectral_bounds(H)
    println("- Spectral bounds calculated")
    mean_E, ΔE = transform_band_center_and_width(E_max, E_min)
    rescale_hamiltonian!(H, mean_E, ΔE)
    println("- Hamiltonian rescaled")

    dim = size(H, 1)
    n_labels = length(unique_labels)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    rank_size = MPI.Comm_size(comm)
    println("- rank $rank / $rank_size")

    BLAS.set_num_threads(1)

    num_vecs = floor(Int, N / rank_size)

    if rank == (rank_size - 1)
        mod_vecs = N % rank_size
        if mod_vecs != 0
            println("- Number of random vectors not optimal: remainder $mod_vecs")
        end
    end

    arr_coeff_COHPs = zeros(Float64, M, n_labels, n_labels, num_vecs)
    coeff_COHPs = zeros(Float64, M, n_labels, n_labels)

    ### get COHP coefficients
    tforeach(1:num_vecs; chunksize=1) do i

        vec = draw_vec(i, dim, rank, num_vecs)
        arr_coeff_COHPs[:, :, :, i] = kernel_polynomial_method_cohp(H, vec, M, basis_labels, unique_labels)

    end

    for m in 1:M
        for i in 1:n_labels
            for j in 1:n_labels
                coeff_COHPs[m, i, j] = reduce(+, arr_coeff_COHPs[m, i, j, :]) ./ num_vecs
            end
        end
    end
    
    MPI.Barrier(comm)
    coeff_COHPs = MPI.Reduce(coeff_COHPs / rank_size, +, comm) 

    ### compute COHPs from coefficients coeff_COHP_m
    if rank == 0

        E_grid, COHPs = compute_cohp(M, mean_E, ΔE, coeff_COHPs, unique_labels)

        for (pair, cohp) in COHPs
            elem_i, elem_j = pair
            output_dir_ij = joinpath(output_dir, "$(elem_i)_$(elem_j)/")
            data_file = joinpath(output_dir_ij, "COHP_$t.txt")
            mkpath(output_dir_ij)
            open(data_file, "w") do io
                println(io, "# Energy   COHP")
                writedlm(io, hcat(E_grid, cohp))
            end
        end
        
    end

    MPI.Finalize()

end


### ### ### ### ### ### ###
### main script execution
### ### ### ### ### ### ###

H_path            = ARGS[1]
output_dir        = ARGS[2]
t                 = parse(Int, ARGS[3])
M                 = parse(Int, ARGS[4])
N                 = parse(Int, ARGS[5])
hamiltonian_style = ARGS[6]


KPM_COHP(H_path, output_dir, t, M, N)

println("KPM COHP calculation for snapshot $t completed.")