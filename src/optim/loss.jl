const loss_to_n = Dict{String, Int64}("MAE" => 1, "MSE" => 2)

"""
    struct Loss

A structure representing a (weighted) loss function for evaluating the difference between predicted and true values. The loss is calculated based on a given norm, with optional weights applied to the errors along the prediction and observation dimensions.

# Fields
- `wE::Vector{Float64}`: A vector of weights applied to the error terms along the `y` dimension (true values).
- `wk::Vector{Float64}`: A vector of weights applied to the error terms along the `ŷ` dimension (predicted values).
- `N::Float64`: Normalization factor, typically the product of the sums of `wE` and `wk`.
- `n::Int64`: The order of the norm used for the loss function (e.g., 1 for L1 norm, 2 for L2 norm, etc.).
"""
mutable struct Loss
    wE :: Vector{Float64}
    wk :: Vector{Float64}
    N :: Float64
    wStr :: Float64
    n :: Int64
    offset :: Float64
    offset_flag :: Bool
    pc_weight :: Float64
    system :: String
    N_VBM :: Int64
    min_delta :: Float64
    gap_weight :: Float64
end

Loss(wE::Vector{Float64}, wk::Vector{Float64}, n::Int64, offset::Float64, offset_flag::Bool, pc_weight::Float64, system::String, N_VBM :: Int64, min_delta :: Float64, gap_weight :: Float64) = Loss(wE, wk, sum(wE)*sum(wk), 1., n, offset, offset_flag, pc_weight, system, N_VBM, min_delta, gap_weight)
Loss(n::Int64) = Loss(Float64[], Float64[], 0, 1., n, 0., true, 1, "default_system", 1, 0.0, 1.0)
"""
    Loss(Nε, Nk; conf=get_empty_config(), loss=get_loss(conf), wE=get_band_weights(conf, Nε), wk=get_kpoint_weights(conf, Nk))

Create a `Loss` object with band weights, k-point weights, and loss norm based on a given configuration.

# Arguments
- `Nε::Int64`: Number of energy bands (dimension of `wE`).
- `Nk::Int64`: Number of k-points (dimension of `wk`).
- `conf`: Configuration object (default: `get_empty_config()`). This is used to retrieve parameters for customizing the loss function.
- `loss`: A string that specifies the type of loss function. The default value is obtained from `get_loss(conf)`.
- `wE`: Vector of weights for the energy bands, defaulted to `get_band_weights(conf, Nε)`.
- `wk`: Vector of weights for the k-points, defaulted to `get_kpoint_weights(conf, Nk)`.

# Keyword Arguments
- `wE`: The band weights, used to scale the errors over energy bands (default provided by configuration).
- `wk`: The k-point weights, used to scale the errors over k-points (default provided by configuration).

# Returns
- A `Loss` object initialized with the appropriate band weights (`wE`), k-point weights (`wk`), normalization factor, and loss norm (`n`).
"""
function Loss(Nε, Nk, conf=get_empty_config(); loss=get_loss(conf), wE=get_band_weights(conf, Nε), wk=get_kpoint_weights(conf, Nk), offset = get_offset(conf))
    n = loss_to_n[loss]

    for key in keys(conf.blocks["Optimizer"])
        if occursin("wk_", key)
            index_range = get_weight_index_from_key(key)
            wk[index_range] .= conf(key, "Optimizer")
        elseif occursin("we_", key)
            index_range = get_weight_index_from_key(key)
            wE[index_range] .= conf(key, "Optimizer")
        end
    end
    return Loss(wE, wk, n,0, offset, 1, 1, 0.0, 1.0)
end

Loss(conf=get_empty_config()) = Loss(loss_to_n[get_loss(conf)])

(l::Loss)(y, ŷ) = forward(l, y, ŷ)

"""
    get_weight_index_from_key(key::String) -> UnitRange{Int64}

Parses a weight key string and returns a range of indices.

# Arguments
- `key::String`: A string representing a weight index, either as a single value (e.g., `"key_1"`) or as a range (e.g., `"key_1-3"`).

# Returns
- A `UnitRange{Int64}` representing the parsed index or index range.
"""
function get_weight_index_from_key(key)
    if occursin('-', key)
        key_split = split(key, '-')
        return parse(Int64, key_split[1][4:end]):parse(Int64, key_split[2])
    else
        index = parse(Int64, key[4:end])
        return index:index
    end
end

"""
    forward(l::Loss, y, ŷ)

Compute the forward pass of the loss function given the true values `y` and the predicted values `ŷ`.

# Arguments
- `l::Loss`: A `Loss` object which specifies how the loss is calculated.
- `y::AbstractArray`: The predicted for each band and k-point.
- `ŷ::AbstractArray`: The true values (ground truth) for each band and k-point.

# Returns
-`L::Float64`: The loss between `y` and `ŷ`.
"""
min_delta = 0.0
function forward(l::Loss, y, ŷ; offset = l.offset, offset_flag = l.offset_flag, min_delta = l.min_delta, gap_weight = l.gap_weight)
    Δy = y - ŷ
    if offset_flag && l.system != "individual"
        Δy = Δy .- offset
    elseif l.system == "individual"
        Δy = Δy .- mean(Δy)
    end

    k_min = argmin(abs.( ŷ[l.N_VBM+1,:] - ŷ[l.N_VBM, :]))
    l.wk[k_min] = gap_weight
    l.N = sum(l.wE)*sum(l.wk)
    #println("wk after $k_min  $(l.wk[k_min])")
    y_mod = abs.(Δy) .+ min_delta
    L_E_avg = vec(mean(y_mod, dims = 2))
    w =  y_mod ./L_E_avg
    #println(maximum(w))
    if isempty(l.wE) && isempty(l.wk)
        return mean(@. abs(Δy)^l.n)^2
    else
        L = @. abs(Δy)^l.n
        L_n = 1/l.N * 1/l.wStr * (l.wE' * (L .* w) * l.wk)
        #println(w)
        return L_n
        #return mean(@. abs(Δy)^l.n)
    end
end

function forward_MAE(l::Loss, y, ŷ; offset = l.offset, offset_flag = l.offset_flag)
    Δy = y - ŷ

    if offset_flag && l.system != "individual"
        Δy = Δy .- offset
    elseif l.system == "individual"
        Δy = Δy .- mean(Δy)
    end

    if isempty(l.wE) && isempty(l.wk)
        return mean(@. abs(Δy))
    else
        return mean(@. abs(Δy))
    end
end

function forward_BG(l::Loss, y, ŷ)
    k_min = argmin(abs.( ŷ[l.N_VBM+1,:] - ŷ[l.N_VBM, :]))
    bg_hat = ŷ[l.N_VBM+1,k_min] - ŷ[l.N_VBM, k_min]
    bg = y[l.N_VBM+1,k_min] - y[l.N_VBM, k_min]
    Δbg = abs(bg - bg_hat)
    return Δbg
end

function forward(l::Loss, Hr::Vector{<:AbstractMatrix}, Ĥrs::Vector{<:AbstractMatrix})
    total_loss = mean([mean(@. abs(H - Ĥ)^l.n) for (H, Ĥ) in zip(Hr, Ĥrs)])
    return total_loss 
end

"""
    backward(l::Loss, y, ŷ)

Compute the gradient of the loss function with respect to the predicted values `ŷ`.

# Arguments
- `l::Loss`: A `Loss` object which specifies how the loss is calculated.
- `y::AbstractVector`: The predicted for each band and k-point.
- `ŷ::AbstractVector`: The true values (ground truth) for each band and k-point.

# Returns
- `dL::AbstractArray`: The gradient of the loss with respect to the predicted values `y`.
"""
function backward(l::Loss, y, ŷ; offset = l.offset, offset_flag = l.offset_flag, min_delta = l.min_delta)
    Δy = y - ŷ

    if offset_flag && l.system != "individual"
        Δy = Δy .- offset
    elseif l.system == "individual"
        Δy = Δy .- mean(Δy)
    end
    y_mod = abs.(Δy) .+ min_delta
    L_E_avg = vec(mean(y_mod, dims = 2))
    w =  y_mod ./L_E_avg

"""    if isempty(l.wE) && isempty(l.wk)
        L_n = mean(@. abs(Δy)^l.n)
    else
        L = @. abs(Δy)^l.n
        L_n = 1/l.N * 1/l.wStr * (l.wE' * (L .* w) * l.wk)
    end"""

    if isempty(l.wE) && isempty(l.wk)
        N = length(y)
        return @. 1/N * sign(Δy) * l.n * abs(Δy)^(l.n - 1) * L_n
    else
        #L = @. abs(Δy)^l.n
        #L_n = 1/l.N * 1/l.wStr * (l.wE' * L * l.wk)
        #return @. 1/l.N * 1/l.wStr * sign(Δy) * l.wk' * l.wE * l.n * abs(Δy)^(l.n - 1) * 2 * sum(L_n)
        return @. 1/l.N * 1/l.wStr * sign(Δy) * l.wk' * l.wE * l.n * abs(Δy)^(l.n - 1) * w
    end
end

function backward(l::Loss, Hr::Vector{<:AbstractMatrix}, Ĥrs::Vector{<:AbstractMatrix})
    N = sum([length(H) for H in Hr])
    dL = map(zip(Hr, Ĥrs)) do (H, Ĥ)
        @. 1/N * sign(H - Ĥ) * l.n * abs(H - Ĥ)^(l.n - 1)
    end
    return dL
end

"""
    struct Regularization

A data structure that defines the regularization parameters used to penalize model complexity during optimization. Regularization is used to penalize large model parameters to avoid overfitting.

# Fields
- `b::Float64`: The barrier for the regularization term. Regularization is only applied if parameter values surpass this barrier.
- `λ::Float64`: The regularization coefficient (lambda), which controls the intensity of regularization. A higher value leads to stronger regularization.
- `n::Int64`: The norm type used for the regularization term, typically corresponding to L1 (`n=1`), L2 (`n=2`), or other norm types.
"""
struct Regularization
    b :: Float64
    λ :: Float64
    n :: Int64
end

"""
    Regularization([conf]; lambda=get_lambda(conf), barrier=get_barrier(conf), lreg=get_lreg(conf)) -> Regularization

Create a `Regularization` struct using configuration parameters from the provided configuration or default values.

# Arguments
- `conf`: A configuration object (optional). Default values are used if not provided.

# Returns
- A `Regularization` struct initialized with the provided or default configuration values for `barrier`, `lambda`, and `lreg`.
"""
function Regularization(conf=get_empty_config(); lambda=get_lambda(conf), barrier=get_barrier(conf), lreg=get_lreg(conf))
    return Regularization(barrier, lambda, lreg)
end

(R::Regularization)(x) = forward(R, x)

"""
    forward(R::Regularization, x::AbstractVector) -> Float64

Compute the regularization penalty for the given input `x` using the regularization parameters 
defined in the `Regularization` struct.

# Arguments
- `R::Regularization`: A `Regularization` object.
- `x::AbstractVector`: The input (parameter) vector over which the regularization penalty is applied.

# Returns
- `Float64`: The total regularization penalty
"""
forward(R::Regularization, x) = R.λ * mapreduce(y -> abs(y) > R.b ? (y-R.b)^R.n : 0., +, x)

"""
    backward(R::Regularization, x::AbstractVector) -> AbstractVector

Compute the gradient of the regularization penalty with respect to the input (parameter) vector `x`.

# Arguments
- `R::Regularization`: A `Regularization` object.
- `x::AbstractVector`: The input (parameter) vector for which the gradient of the regularization penalty is computed.

# Returns
- `AbstractVector`: A vector of the same size as `x`, where each element is the gradient of the penalty
  with respect to the corresponding element in `x`.
"""
backward(R::Regularization, x) = R.λ .* map(y -> abs(y) > R.b ? R.n * (y-R.b)^(R.n-1) : 0., x)


function Losses(Nε_all, Nk_all, N_eig_avg, N_VBM_all, N_weight_all, systems, conf=get_empty_config();weights = false, loss=get_loss(conf), offset_flag = get_offset(conf), offset_mode = get_offset_mode(conf), min_delta = get_min_delta(conf), gap_weight = get_gap_weight(conf))
    N_strc = length(Nk_all)
    n = loss_to_n[loss]
    #n = 2
    #Loss_vec = Vector{Loss}(undef, N_stc)
    Loss_vec = [Loss(n) for i in 1:N_strc]
    #println("N_VBM_all $N_VBM_all")
    #N_eig_avg = sum(Nε_all .* Nk_all)/N_strc
    for i in 1:N_strc
        Nε = Nε_all[i]
        Nk = Nk_all[i]
        pc_weight = N_weight_all[i]
        system = decide_system(systems[i], offset_mode)
        N_VBM = N_VBM_all[i]
        #gap_width = ceil(Int, 0.05 * Nε)
        gap_width = ceil(Int, N_VBM/10)
        SC_size = ceil(Int, N_VBM/10)
        wStr = Nk / N_eig_avg
        wStr = 1

        wE = ones(Nε) * 0.1
        wE[1:gap_width] .= 0.5 
        wE[gap_width+1:N_VBM - gap_width - 1] .= 1 
        #wE[1] = N_VBM_all[i] == 10 ? 1 : 0.1
        #wE[N_VBM - 2 * gap_width : N_VBM - gap_width ] .= 1
        #wE[N_VBM + gap_width + 1 : min(N_VBM + 2 * gap_width + 1, Nε)] .= 1
        #wE[N_VBM - gap_width + 1 : min(N_VBM + gap_width, Nε)] .= 2
        wE[N_VBM - gap_width : min(N_VBM + 1 + gap_width, Nε)] .= 2 
        wE[N_VBM:N_VBM+1] .= sum(wE) * 0.25      # valence band maximum

        #println("gapwidth $gap_width       wE    (   $wE  )")

        wE = get_auto_band_weights(conf) ? wE : ones(Nε) 
        wk = weights ? get_kpoint_weights(conf, Nk) : ones(Nk)

        #wE = weights ? get_band_weights(conf, Nε) : ones(Nε)

        wk_min = gap_weight * sum(wk)
        Loss_vec[i] = Loss(wE, wk, sum(wE)*sum(wk), wStr, n, 0, offset_flag, pc_weight, system, N_VBM, min_delta, wk_min)
    end
    
    return Loss_vec
end

function decide_system(system, offset_mode)
    if startswith(offset_mode, "s")
        result = endswith(system, "_PC") ? system[1:end-3] : system
    elseif startswith(offset_mode, "o")
        result = "one_system"
    elseif startswith(offset_mode, "i")
        result = "individual"
    else
        result = system
    end
    return result
end