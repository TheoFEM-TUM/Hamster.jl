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
    soc=false

The `soc` tag switches on spin-orbit coupling (SOC).
"""
get_soc(conf::Config)::Bool = conf("soc") == "default" ? false : conf("soc")

"""
    sp_tol=1e-10

The `sp_tol` tag sets a tolerance for value to be considered zero.
"""
get_sp_tol(conf::Config)::Float64 = conf("sp_tol") == "default" ? 1e-10 : conf("sp_tol")

"""
    sp_mode=dense

The `sp_mode` tag switches between dense and sparse matrix methods.
"""
function get_sp_mode(conf::Config)::Union{Sparse, Dense}
    if conf("sp_mode") == "default"
        return Dense()
    else
        conf("sp_mode") ? Sparse() : Dense()
    end
end

"""
    sp_diag=dense

The `sp_diag` tag switches between dense and sparse methods for matrix diagonalization.
"""
function get_sp_diag(conf::Config)::Union{Sparse, Dense}
    if conf("sp_mode") == "default"
        return Dense()
    else
        conf("sp_mode") ? Sparse() : Dense()
    end
end