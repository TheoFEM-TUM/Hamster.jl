using SparseArrays, LinearAlgebra, KrylovKit


### get spectral bounds of the hamiltonian
function get_spectral_bounds(hamiltonian::SparseMatrixCSC{ComplexF64})
   
    E_max = real(eigsolve(hamiltonian, 1, :LR, ishermitian=true)[1][1])::Float64
    E_min = real(eigsolve(hamiltonian, 1, :SR, ishermitian=true)[1][1])::Float64

    return E_max, E_min
end

### transform band center and width to mean and half-width
function transform_band_center_and_width(E_max, E_min)

    ϵ = 0.01
    mean_E = (E_min .+ E_max) ./ 2.0
    ΔE = (E_max .- E_min) ./ (2.0 .- ϵ)

    return mean_E, ΔE
end

### rescale hamiltonian so that spectrum is [-1;1]
function rescale_hamiltonian!(hamiltonian::SparseMatrixCSC{ComplexF64}, mean_E::Float64, ΔE::Float64)
    for i in axes(hamiltonian, 1)
        hamiltonian[i, i] -= mean_E  # Subtract mean_E only from diagonal elements
    end
    hamiltonian ./= ΔE
end

function rescale_energy(E::Float64, mean_E::Float64, ΔE::Float64)
    return (E .- mean_E) ./ ΔE
end

### analytic function for chebyshev polynomial
function chebyshev_polynomials(x, m::Int)
    return cos.(m .* acos.(x))
end

### return Jackson kernel for specific m
function jackson_kernel_elem(m::Int, M::Int)
    g_m = ((M + 1 - m) * cos(m * pi / (M + 1)) + sin(m * pi / (M + 1)) * cot(pi / (M + 1))) / (M + 1)
    return g_m
end


### construction of vectors for trace calculation 
function draw_vec(i::Int, dim::Int, rank::Int, num_vecs::Int)
        
    Random.seed!(1234 + i + rank * num_vecs)

    vec = exp.(1im .* rand(Uniform(0.0, 2 * pi), dim))

    vec *= sqrt(dim)/norm(vec)

    return vec
end


function get_delta_m_E(M::Int, mean_E::Float64, ΔE::Float64, g_m::Vector{Float64})

    z = 2 * M

    delta_m_E = zeros(Float64, z, M)

    E = LinRange(-0.999, 0.999, z)

    for m in 1:M

        if m - 1 == 0
            factor = 1
        else
            factor = 2
        end

        delta_m_E[:, m] .= g_m[m] * factor./ΔE .* chebyshev_polynomials(E, m-1) ./ (pi * sqrt.(1 .- E.^2))

    end

    E = ΔE .* E .+ mean_E 

    return E, delta_m_E

end