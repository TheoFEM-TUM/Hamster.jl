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

"""
    tb_model=true

The `tb_model` tag switches on the use of a TB model in the effective Hamiltonian model.
"""
get_tb_model(conf::Config)::Bool = conf("tb_model") == "default" ? true : conf("tb_model")

"""
**sp_tol**=1e-10

The `sp_tol::Float` tag sets a tolerance for values to be considered zero.
"""
get_sp_tol(conf::Config)::Float64 = conf("sp_tol") == "default" ? 1e-10 : conf("sp_tol")

"""
**sp_mode**=false

The `sp_mode::Bool` tag switches between dense and sparse matrix methods. 
This will only affect the computation of Hᴿ (not Hᵏ or diagonalization) and gradient computations when doing optimization.
"""
function get_sp_mode(conf::Config)::Union{Sparse, Dense}
    if conf("sp_mode") == "default"
        return Dense()
    else
        conf("sp_mode") ? Sparse() : Dense()
    end
end

"""
**sp_diag**=false

The `sp_diag::Bool` tag switches between dense and sparse methods for matrix diagonalization (only affects Hᵏ and diagonalization).
This can not be combined with optimization.
"""
function get_sp_diag(conf::Config)::Union{Sparse, Dense}
    if conf("sp_diag") == "default"
        return Dense()
    else
        conf("sp_diag") ? Sparse() : Dense()
    end
end

"""

"""
get_target_directory(conf::Config)::String = conf("target_directory") == "default" ? "missing" : conf("target_directory")


