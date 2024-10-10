abstract type AbstractLoss<:Function end

(l::AbstractLoss)(y, ŷ) = forward(l, y, ŷ)
"""
    (mse::MSE)(y, ŷ)

Calculate the mean-square error of predictions y given the ground truth ŷ.
"""
struct MSE <: AbstractLoss; end
forward(::MSE, y, ŷ) = mean((y .- ŷ).^2)
gradient(::MSE, y, ŷ) = 1 ./ prod(size(y)) .* 2 .*(y .- ŷ)

"""

    (mae::MAE)(y, ŷ)

Calculate the mean-absolute error of predictions y given the ground truth ŷ.
"""
struct MAE <: AbstractLoss; end
forward(mae::MAE, y, ŷ) = mean(@. abs(y - ŷ))
gradient(mae::MAE, y, ŷ) = 1 ./ prod(size(y)) .* @. ifelse(y - ŷ > 0, 1, -1)

"""
    (nmse::nMSE)(y, ŷ)

Calculate the mean-squared error of predictions `y` given the ground truth `ŷ` and normalize the result by the
absolute value of `ŷ`.
"""
struct nMSE <: AbstractLoss; end
forward(nmse::nMSE, y, ŷ) = mean(@. (y - ŷ)^2 / abs(ŷ))
gradient(nmse::nMSE, y, ŷ) = @. 2 * (y - ŷ) / abs(ŷ)

"""
    (nmae::nMAE)(y, ŷ)

    Calculate the mean-absolute error of predictions `y` given the ground truth `ŷ` and normalize the result by the
    absolute value of `ŷ`.
"""
struct nMAE <: AbstractLoss; end
forward(nmae::nMAE, y, ŷ) = mean(@. abs(y - ŷ) / abs(ŷ))
gradient(nmae::nMAE, y, ŷ) = @. ifelse(y - ŷ > 0, 1, -1) / abs(ŷ)

"""

    (wmse::wMSE)(ε, Ê)

Calculate the weighted mean-square error of the energy band predictions `ε`
given the ground truth `Ê` with band weights `wε` and k-point weigths `wk`.
"""
struct wMSE <: AbstractLoss
    wk :: Array{Float64, 1}
    wε :: Array{Float64, 1}
end

function forward(l::wMSE, ε, Ê)
    L = 1/(sum(l.wk) * sum(l.wε))*( (ε .- Ê).^2 * l.wk ) ⋅ l.wε
end
function forward(l::wMSE, ε::A31, Ê::A32) where {A31, A32 <: AbstractArray{Float64, 3}}
    N = size(Ê, 3); L = 0
    for n in 1:N
        L += forward(l, ε[:, :, n], Ê[:, :, n])
    end
    return L ./ N
end

function gradient(l::wMSE, ε, Ê)
    dL = 1/(sum(l.wk) * sum(l.wε))*(2 .*(ε .- Ê) .* l.wk' .* l.wε)
end

function gradient(l::wMSE, ε::A31, Ê::A32) where {A31, A32 <: AbstractArray{Float64, 3}}
    N = size(Ê, 3); dL = similar(ε)
    for n in 1:N
        dL[:, :, n] = gradient(l, ε[:, :, n], Ê[:, :, n])
    end
    return dL ./ N
end

"""

    (wmse::wMAE)(ε, Ê)

Calculate the weighted mean-absolute error of the energy band predictions `ε`
given the ground truth `Ê` with band weights `wε` and k-point weigths `wk`.
"""
struct wMAE <: AbstractLoss
    wk :: Array{Float64, 1}
    wε :: Array{Float64, 1}
end

function forward(l::wMAE, ε, Ê)
    L = 1/(sum(l.wk) * sum(l.wε))*( abs.(ε .- Ê) * l.wk ) ⋅ l.wε
end
function forward(l::wMAE, ε::A31, Ê::A32) where {A31, A32 <: AbstractArray{Float64, 3}}
    N = size(Ê, 3); L = 0
    for n in 1:N
        L += forward(l, ε[:, :, n], Ê[:, :, n])
    end
    return L ./ N
end

function gradient(l::wMAE, ε, Ê)
    dL = 1/(sum(l.wk) * sum(l.wε))*(ifelse.(ε .> Ê, 1, -1) .* l.wk' .* l.wε)
end

function gradient(l::wMAE, ε::A31, Ê::A32) where {A31, A32 <: AbstractArray{Float64, 3}}
    N = size(Ê, 3); dL = similar(ε)
    for n in 1:N
        dL[:, :, n] = gradient(l, ε[:, :, n], Ê[:, :, n])
    end
    return dL ./ N
end

abstract type AbstractRegularization; end

struct L1Regularization <: AbstractLoss
    λ :: Real
end

forward(Rl1::L1Regularization, x) = Rl1.λ * sum(x)
gradient(Rl1::L1Regularization, x) = Rl1.λ * ones(x)

struct L2Regularization{R<:Real} <: AbstractRegularization
    λ :: R
end

forward(Rl2::L2Regularization, x) = Rl2.λ * sum(x.^2)
gradient(Rl2::L2Regularization, x) = Rl2.λ .* x

struct L2Barrier{R1,R2<:Real} <: AbstractRegularization
    b :: R1
    λ :: R2
end

(f::L2Barrier)(x) = f.λ*mapreduce(y -> abs(y) > f.b ? (f.b-y)^2 : 0., +, x)
gradient(f::L2Barrier, x) = f.λ .* map(y -> abs(y) > f.b ? 2*(y-f.b) : 0, x)

const loss_dict = Dict{String, Any}("MSE"=>MSE, "MAE"=>MAE, "wMAE"=>wMAE, "wMSE"=>wMSE, "nMSE"=>nMSE, "nMAE"=>nMAE)

"""
    init_loss(data, conf)

Obtain the loss function from `conf`.
"""
function init_loss(data::Tuple, l_conf::AbstractString, conf)
    Nε, Nk = size(data[2])
    if l_conf[1] == 'w'
        wε_conf = conf("wE", "Optimizer")
        wk_conf = conf("wk", "Optimizer")
        wε = ifelse(wε_conf=="default", ones(Nε), wε_conf)
        wk = ones(Nk); if wk_conf ≠ "default"; wk = wk .* wk_conf; end
        for key in keys(conf("Optimizer"))
            if occursin("wk_", key)
                index = parse(Int64, key[4:end])
                wk[index] = conf(key, "Optimizer")
            end
        end
        return loss_dict[l_conf](wk, wε)
    else
        return loss_dict[l_conf]()
    end
end

init_loss(data::Tuple, conf::TBConfig) = init_loss(data, get_loss(conf), conf)

"""
    get_regularization(constants, conf)

Read the specified regularization function from the config file.
"""
function get_regularization(conf::TBConfig)
    λ = get_lambda(conf)
    b = get_barrier(conf)
    if conf("barrier", "Optimizer") == "default"
        return L2Regularization(λ)
    else
        return L2Barrier(b, λ)
    end
end