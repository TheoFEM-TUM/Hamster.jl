"""
    get_exp_ikR(k⃗, R⃗)
    
Calculate the phase factor exp(2πik⃗⋅R⃗).
"""
@inline exp_2πi(R⃗, k⃗) = @. exp(2π*im * $*(R⃗', k⃗))

"""
    get_hamiltonian(Hᴿ, Rs, ks)

Calculate the Hamiltonian for the real-space Hamiltonian matrices `Hᴿ` at grid
points `Rs` and the k-points `ks`.
"""
function get_hamiltonian(Hᴿ::Array{Float64, 3}, Rs, ks)
    Nε = size(Hᴿ, 1); Nk = size(ks, 2)
    Hᵏ = zeros(ComplexF64, Nε, Nε, Nk); Hᴿ = complex.(Hᴿ)
    exp_2πiRk = exp_2πi(Rs, ks)
    @tensor Hᵏ[i, j, k] = Hᴿ[i, j, R] * exp_2πiRk[R, k]
    return Hᵏ
end

function get_hamiltonian(Hr::Vector{<:AbstractMatrix}, Rs, ks)
    Nε = size(Hr[1], 1)
    Hk = [spzeros(ComplexF64, Nε, Nε) for _ in axes(ks, 2)]
    exp_2πiRk = exp_2πi(Rs, ks)

    @views for k in eachindex(Hk)
        Hk[k] = sum(Hr .* exp_2πiRk[:, k])
    end
    return Hk
end