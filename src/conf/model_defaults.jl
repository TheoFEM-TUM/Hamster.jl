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
function get_sp_mode(conf::Config)::Union{Type{Val{:sparse}}, Type{Val{:dense}}}
    if conf("sp_mode") == "default"
        return Val{:dense}
    else
        conf("sp_mode") ? Val{:sparse} : Val{:dense}
    end
end

"""
    sp_diag=dense

The `sp_diag` tag switches between dense and sparse methods for matrix diagonalization.
"""
function get_sp_diag(conf::Config)::Union{Type{Val{:sparse}}, Type{Val{:dense}}}
    if conf("sp_diag") == "default"
        return Val{:dense}
    else
        conf("sp_diag") ? Val{:sparse} : Val{:dense}
    end
end