
function broadening_weight(dE::Float64, sigma::Float64, kind::Symbol)::Float64
    if kind === :gaussian
        return exp(-0.5 * (dE / sigma)^2) / (sigma * sqrt(2π))
    elseif kind === :lorentzian
        return (sigma / π) / (dE^2 + sigma^2)
    else
        throw(ArgumentError("Unknown broadening kind: $kind. Use :gaussian or :lorentzian."))
    end
end

function projector(v::Vector{ComplexF64}, p::Vector{Float64})
    w = copy(v)
    for i in eachindex(p)
        if p[i] == 0.0
            w[i] = 0.0
        end
    end
    return w
end

function get_projectors(basis_labels, unique_labels)
    
    n_labels = length(unique_labels)
    dim = length(basis_labels)
    Ps = zeros(Float64, dim, n_labels)

    for i in 1:n_labels
        for j in 1:dim
            if basis_labels[j] == unique_labels[i] 
                Ps[j,i] = 1
            end
        end
    end

    return Ps
end
