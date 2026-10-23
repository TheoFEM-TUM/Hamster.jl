#
# Orbital / basis config
#
@configtag onsite Bool true "use separate parameters for onsite matrix elements." category="basis"
@configtag sepNN Bool false "use separate parameters for nearest-neighbor interactions. This can improve accuracy, however, may at the same time affect transferability negatively." category="basis"

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
@configtag orbitals Vector{String} String[] "orbitals of the species (e.g. `s px py pz`, `sp3dr2 ...`); without orbitals, the species does not enter the electronic structure (but its charge enters the Ewald model)." "atom" per_atom=true getter=:custom default_doc="(empty)" category="basis"

@configtag itp_xmin Float64 0. "minimum x value for interpolation." category="basis"
@configtag itp_xmax Float64 get_rcut(conf)+abs(get_rcut_tol(conf))+1 "maximum x value for interpolation." category="basis" default_doc="rcut + |rcut_tol| + 1"
@configtag itp_Ninit Int64 20 "number of initial interpolation points." category="basis"
@configtag itp_Nmax Int64 300 "maximum number of interpolation points." category="basis"
@configtag itp_tol Float64 1e-5 "convergence criterion for interpolation." category="basis"
@configtag load_rllm Bool false "read rllm data from a file." category="basis" see_also=["rllm_file"]
@configtag interpolate_rllm Bool true "force interpolation of rllm." category="basis"

"""
**rllm_file**=rllm.dat

The `rllm_file::String` tag sets the name of the file where the distance dependence is stored.
"""
function get_rllm_file(conf::Config)::String 
    if conf("rllm_file") == "default" 
        return length(get_systems(conf)) > 1 ? "rllm.h5" : "rllm.dat"
    else
        return conf("rllm_file")
    end
end
@configtag rllm_file String "rllm.dat" "path to the file with the distance dependence of the matrix elements (rllm)." getter=:custom default_doc="rllm.dat (rllm.h5 for several systems)" category="basis" see_also=["load_rllm"]

"""
**tmethod**=rotation

The `tmethod::String` tag selects the method used to compute the Slater-Koster transformation matrix.  
In practice, this choice should not affect the results, although `rotation` is slightly more efficient.

Possible options:
- `rotation`: constructs the new reference system from the rotation matrix (default).
- `gramschmidt`: constructs the new reference system using a Gram-Schmidt procedure.
"""
get_tmethod(conf::Config)::String = conf("tmethod") == "default" ? "rotation" : conf("tmethod")
@configtag tmethod String "rotation" "method for the Slater-Koster transformation matrix (rotation or gramschmidt; the results are the same)." getter=:custom category="basis"