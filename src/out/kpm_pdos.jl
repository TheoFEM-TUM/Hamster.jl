using SparseArrays, LinearAlgebra, KrylovKit
using DelimitedFiles
using Base.Threads: nthreads
using Random, Statistics, Distributions
using OhMyThreads, MPIPreferences
using MPI
include("../KPM.jl")
include("../helper.jl")



### calculate coeff_PDOS_m = v * P_i * T_m(H) * v
function kernel_polynomial_method_pdos(H::SparseMatrixCSC{ComplexF64}, v::Vector{ComplexF64}, M::Int, basis_labels::Vector{String}, unique_labels::Vector{String})

    n_labels = length(unique_labels)

    Ps = get_projectors(basis_labels, unique_labels)

    coeff_PDOS = Array{Float64}(undef, M, n_labels)

    v1 = copy(v) 
    v2 = H * v
    v3 = zeros(ComplexF64, length(v))

    pv = [projector(v, Ps[:, i]) for i in 1:n_labels]

    for i in 1:n_labels
        coeff_PDOS[1, i] = real(pv[i] ⋅ v1)
        coeff_PDOS[2, i] = real(pv[i] ⋅ v2)
    end

    for m in 3:M

        v3 .= 2 .* H * v2 .- v1

        for i in 1:n_labels
            coeff_PDOS[m, i] = real(pv[i] ⋅ v3)
        end

        v1 .= v2
        v2 .= v3

    end

    return coeff_PDOS
end

### compute PDOS from coefficients coeff_PDOS_m
function compute_pdos(M, mean_E, ΔE, coeff_PDOS, unique_labels)

    n_labels = length(unique_labels)

    g_m = [jackson_kernel_elem(m, M) for m in 1:M]

    E_grid, delta_m_E = get_delta_m_E(M, mean_E, ΔE, g_m)

    PDOSs = Dict{String, Vector{Float64}}()

    for i in 1:n_labels
        label = unique_labels[i]
        PDOSs[label] = zeros(Float64, 2 * M)
        for m in 1:M
            PDOSs[label] .+= coeff_PDOS[m, i] * delta_m_E[:, m]
        end
    end

    return E_grid, PDOSs
end

### main function to run KPM-PDOS calculation
function KPM_PDOS(H_path::String, output_dir::String, t::Int, M::Int, N::Int)

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

    arr_coeff_PDOSs = zeros(Float64, M, n_labels, num_vecs)
    coeff_PDOSs = zeros(Float64, M, n_labels)

    ### get PDOS coefficients
    tforeach(1:num_vecs; chunksize=1) do i

        vec = draw_vec(i, dim, rank, num_vecs)
        arr_coeff_PDOSs[:, :, i] = kernel_polynomial_method_pdos(H, vec, M, basis_labels, unique_labels)

    end

    for m in 1:M
        for i in 1:n_labels
            coeff_PDOSs[m, i] = reduce(+, arr_coeff_PDOSs[m, i, :]) ./ num_vecs
        end
    end

    MPI.Barrier(comm)
    coeff_PDOSs = MPI.Reduce(coeff_PDOSs / rank_size, +, comm)

    ### compute PDOS from coefficients
    if rank == 0

        E_grid, PDOSs = compute_pdos(M, mean_E, ΔE, coeff_PDOSs, unique_labels)

        for (label, pdos) in PDOSs
            output_dir_i = joinpath(output_dir, "$(label)/")
            data_file = joinpath(output_dir_i, "PDOS_$t.txt")
            mkpath(output_dir_i)
            open(data_file, "w") do io
                println(io, "# Energy   PDOS")
                writedlm(io, hcat(E_grid, pdos))
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


KPM_PDOS(H_path, output_dir, t, M, N)

println("KPM PDOS calculation for snapshot $t completed.")