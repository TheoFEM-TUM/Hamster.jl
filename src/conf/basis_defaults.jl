"""
    onsite=true

The `onsite` tag switches on the use of an extra parameter set for onsite interactions.
"""
get_onsite(conf::Config)::Bool = conf("onsite") == "default" ? true : conf("onsite")

"""
    sepNN=false

The `sepNN` tag switches on the use of an extra parameter set for nearest-neighbor interactions (compared to further away interactions).
"""
get_sepNN(conf::Config)::Bool = conf("sepNN") == "default" ? false : conf("sepNN")

"""
    alpha=0.7*Z

The `alpha` value determines how rapid the orbital overlap for a specific ion type falls off with distance. Defaults to 70% of the core charge.
"""
get_alpha(conf::Config, type)::Float64 = conf("alpha", type) == "default" ? 0.70 * elements[Symbol(type)].number : conf("alpha", type)

"""
    n=n_period

The `n` value determines the order of the polynomial that is used to model the distance dependence the orbital overlap for a specific ion type. Defaults to the period the atom species belongs to.
"""
get_n(conf::Config, type)::Int64 = conf("n", type) == "default" ? elements[Symbol(type)].period : conf("n", type)

"""
    NNaxes=false

If `NNaxes=true` the orbital axes are rotated along the connecting vectors with the nearest neighbors of the respective orbitals. The number of nearest neighbors depends on the number of orbitals.
"""
get_nnaxes(conf::Config, type)::Bool = conf("NNaxes", type) == "default" ? false : conf("NNaxes", type)

"""
    orbitals=[s, px, py, pz]

The `orbitals` tag defines the set of orbitals that is used as a basis for the ion of `type`.
"""
function get_orbitals(conf::Config, type)::Vector{String}
    if conf("orbitals", type) == "default" 
        return String[]
    else
        # Convert orbitals to a Vector if it is not
        if typeof(conf("orbitals", type)) <: AbstractVector 
            return conf("orbitals", type) 
        else
            return [conf("orbitals", type)]
        end
    end
end

"""
    itp_xmin=0.

The `itp_xmin` tag sets the minimal x value used for the adaptive interpolation.
"""
get_itp_xmin(conf::Config)::Float64 = conf("itp_xmin") == "default" ? 0. : conf("itp_xmin")

"""
    itp_xmax=rcut+1

The `itp_xmax` tag sets the maximal x value used for the adaptive interpolation.
"""
get_itp_xmax(conf::Config)::Float64 = conf("itp_xmax") == "default" ? get_rcut(conf)+2 : conf("itp_xmax")

"""
    itp_Ninit=10

The `itp_Ninit` tag sets the number of initial points used for the adaptive interpolation.
"""
get_itp_Ninit(conf::Config)::Int64 = conf("itp_Ninit") == "default" ? 20 : conf("itp_Ninit")

"""
    itp_Nmax=1000

The `itp_Nmax` tag sets the maximum number of points used for the adaptive interpolation.                            
"""
get_itp_Nmax(conf::Config)::Int64 = conf("itp_Nmax") == "default" ? 300 : conf("itp_Nmax")

"""
    itp_tol=1e-7

The `itp_tol` tag sets the numerical tolerance for the adaptive interpolation
"""
get_itp_tol(conf::Config)::Float64 = conf("itp_tol") == "default" ? 1e-5 : conf("itp_tol")

"""
    load_rllm=false

The `load_rllm` tag decides whether the distance dependence is read from a file.
"""
get_load_rllm(conf::Config)::Bool = conf("load_rllm") == "default" ? false : conf("load_rllm")

"""
    interpolate_rllm=true

The `interpolate_rllm` tag switches on interpolation of the distance dependence of overlaps.
"""
get_interpolate_rllm(conf::Config)::Bool = conf("interpolate_rllm") == "default" ? true : conf("interpolate_rllm")

"""
    rllm_file=rllm.dat

The `rllm_file` tag sets the name of the file where the distance dependence is stored.
"""
get_rllm_file(conf::Config)::String = conf("rllm_file") == "default" ? "rllm.dat" : conf("rllm_file")

"""
    tmethod=Rotation

The `tmethod` tag sets the method that is used to calculate the Slater-Koster transform matrix.
"""
get_tmethod(conf::Config)::String = conf("tmethod") == "default" ? "Rotation" : conf("tmethod")