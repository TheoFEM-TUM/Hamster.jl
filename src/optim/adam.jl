# Implementation adapted from https://github.com/FluxML/Flux.jl

const EPS = 1e-8

"""
    Adam(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $EPS)
[Adam](https://arxiv.org/abs/1412.6980) optimiser.
# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
# Examples
```julia
opt = Adam()
opt = Adam(0.001, (0.9, 0.8))
```
"""
mutable struct Adam
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
Adam(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = EPS) = Adam(η, β, ϵ, IdDict())
Adam(η::Real, β::Tuple, state::IdDict) = Adam(η, β, EPS, state)

function apply!(o::Adam, x, Δ)
  η, β = o.eta, o.beta

  mt, vt, βp = get!(o.state, x) do
      (zero(x), zero(x), Float64[β[1], β[2]])
  end :: Tuple{typeof(x),typeof(x),Vector{Float64}}

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
  βp .= βp .* β

  return Δ
end

function update!(opt::Adam, x::AbstractArray, x̄)
    x̄r = copyto!(similar(x̄), x̄)

    x .-= apply!(opt, x, x̄r)
  end