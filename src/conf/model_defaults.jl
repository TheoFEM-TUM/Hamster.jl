"""
Abstract type `SparsityMode` serves as a base for indicating the sparsity of matrices.
These types enable dispatching based on the sparsity mode.

# Subtypes
- `Sparse <: SparsityMode`: Represents matrices with sparse storage.
- `Dense <: SparsityMode`: Represents matrices with dense storage.
"""
abstract type SparsityMode end
struct Sparse<:SparsityMode end
struct Dense<:SparsityMode end

@configtag tb_model Bool true "build TB model."
@configtag sp_tol Float64 1e-10 "tolerance for value to be considered zero."

"""
**sp_mode**=false

The `sp_mode::Bool` tag switches between dense and sparse matrix methods. 
This will only affect the computation of Hᴿ (not Hᵏ or diagonalization) and gradient computations when doing optimization.
"""
function get_sp_mode(conf::Config)::Union{Sparse, Dense}
    if conf("sp_mode") == "default"
        return Sparse()
    else
        conf("sp_mode") ? Sparse() : Dense()
    end
end
push!(CONFIG_TAGS, ConfigTag{Bool}("sp_mode", conf->get_sp_mode(conf) isa Sparse, "switches to sparse matrix methods (only affects Hᴿ and gradients)."))

"""
**sp_diag**=false

The `sp_diag::Bool` tag switches between dense and sparse methods for matrix diagonalization (only affects Hᵏ and diagonalization).
This can not be combined with optimization.
"""
function get_sp_diag(conf::Config)::Union{Sparse, Dense}
    if conf("sp_diag") == "default" && !get_skip_diag(conf)
        return Dense()
    elseif conf("sp_diag") == "default" && get_skip_diag(conf)
        return get_sp_mode(conf)
    else
        conf("sp_diag") ? Sparse() : Dense()
    end
end
push!(CONFIG_TAGS, ConfigTag{Bool}("sp_diag", conf->get_sp_diag(conf) isa Sparse, "switches to sparse matrix diagonalization (only affects Hᵏ and diagonalization)."))