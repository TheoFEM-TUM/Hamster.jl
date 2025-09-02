"""
**onsite**=true

The `onsite` tag switches on the use of an extra parameter set for onsite interactions.
If false, onsite matrix elements are computed evaluating the distance-dependence function at `r=0`.
"""
get_onsite(conf::Config)::Bool = conf("onsite") == "default" ? true : conf("onsite")

"""
**sepNN**=false

The `sepNN::Bool` tag switches on the use of an extra parameter set for nearest-neighbor interactions (compared to further away interactions).
This can improve accuracy, however, may at the same time affect transferability negatively.
"""
get_sepNN(conf::Config)::Bool = conf("sepNN") == "default" ? false : conf("sepNN")

"""
**orbitals**=[]

The `orbitals` tag specifies the set of orbitals used as the basis for the ion of the given `type`.  
If left empty, no orbitals are assigned and the corresponding atomic species is excluded from the electronic structure.
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
**itp_xmin**=0.

The `itp_xmin::Float` tag sets the minimal x value used for the adaptive interpolation.
"""
get_itp_xmin(conf::Config)::Float64 = conf("itp_xmin") == "default" ? 0. : conf("itp_xmin")

"""
**itp_xmax**=rcut+rcut_tol+1

The `itp_xmax::Float` tag sets the maximal x value used for the adaptive interpolation.
"""
get_itp_xmax(conf::Config)::Float64 = conf("itp_xmax") == "default" ? get_rcut(conf)+abs(get_rcut_tol(conf))+1 : conf("itp_xmax")

"""
**itp_Ninit**=20

The `itp_Ninit::Int` tag sets the number of initial points used for the adaptive interpolation.
"""
get_itp_Ninit(conf::Config)::Int64 = conf("itp_Ninit") == "default" ? 20 : conf("itp_Ninit")

"""
**itp_Nmax**=1000

The `itp_Nmax::Int` tag sets the maximum number of points used for the adaptive interpolation.                            
"""
get_itp_Nmax(conf::Config)::Int64 = conf("itp_Nmax") == "default" ? 300 : conf("itp_Nmax")

"""
**itp_tol**=1e-5

The `itp_tol::Float` tag sets the numerical tolerance for the adaptive interpolation.
"""
get_itp_tol(conf::Config)::Float64 = conf("itp_tol") == "default" ? 1e-5 : conf("itp_tol")

"""
**load_rllm**=false

The `load_rllm::Bool` tag decides whether the distance dependence is read from a file.
"""
get_load_rllm(conf::Config)::Bool = conf("load_rllm") == "default" ? false : conf("load_rllm")

"""
**interpolate_rllm**=true

The `interpolate_rllm::Bool` tag switches on interpolation of the distance dependence of overlaps.
"""
get_interpolate_rllm(conf::Config)::Bool = conf("interpolate_rllm") == "default" ? true : conf("interpolate_rllm")

"""
**rllm_file**=rllm.dat

The `rllm_file::String` tag sets the name of the file where the distance dependence is stored.
"""
get_rllm_file(conf::Config)::String = conf("rllm_file") == "default" ? "rllm.dat" : conf("rllm_file")

"""
**tmethod**=rotation

The `tmethod::String` tag selects the method used to compute the Slater-Koster transformation matrix.  
In practice, this choice should not affect the results, although `rotation` is slightly more efficient.

Possible options:
- `rotation`: constructs the new reference system from the rotation matrix (default).
- `gramschmidt`: constructs the new reference system using a Gram-Schmidt procedure.
"""
get_tmethod(conf::Config)::String = conf("tmethod") == "default" ? "rotation" : conf("tmethod")