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

@configtag tb_model Bool true "build TB model." category="model"
@configtag sp_tol Float64 1e-10 "tolerance for value to be considered zero." category="model"

"""
**sp_mode**=true

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
@configtag sp_mode Bool true "sparse matrices for Hᴿ and the gradients of fits (not for Hᵏ and the diagonalization, see sp_diag)." getter=:custom value=(conf -> get_sp_mode(conf) isa Sparse) category="model" see_also=["sp_diag"]

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
@configtag sp_diag Bool false "sparse Hᵏ and sparse eigensolvers (only some eigenvalues, see diag_method, neig and eig_target); not in fits." getter=:custom value=(conf -> get_sp_diag(conf) isa Sparse) default_doc="false (with skip_diag = true: same as sp_mode)" category="diagonalization" see_also=["diag_method", "neig", "eig_target"]