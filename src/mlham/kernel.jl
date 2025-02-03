mutable struct HamiltonianKernel{T1, T2}
    ws :: Vector{Float64}
    xs :: Vector{T1}
    sim_params :: T2
end

(k::HamiltonianKernel)(xin) = mapreduce(wx->wx[1]*exp_sim(wx[2], xin, σ=k.sim_params), +, zip(k.ws, k.xs))

exp_sim(x₁, x₂; σ=0.1)::Float64 = exp(-normdiff(x₁, x₂)^2 / σ)