"""
    fcut(r, rcut)

Cut-off function whose value smoothly transitions to zero as `r` approaches `rcut`.
Ensures continuity by using a cosine-based smoothing function.

# Arguments
- `r`: The input distance.
- `rcut`: The cutoff radius beyond which the function returns zero.
- `rcut_tol`: A tolerance applied to the cut-off radius, can be positive or negative.

# Returns
- A smoothly varying value between 1 and 0, with `fcut(r, rcut) = 0` for `r > rcut`.
"""
function fcut(r, rcut)
    if r > rcut
        return 0.0
    elseif rcut ≠ 0
        return 1/2 * (cos(π*r/rcut) + 1)
    else
        return 1.0
    end
end

function fcut(r, rcut, rcut_tol)
    if rcut_tol > 0 && r > rcut
        return fcut(r - rcut, rcut_tol)
    elseif rcut_tol < 0 && (rcut - abs(rcut_tol) ≤ r ≤ rcut)
        return fcut(r-rcut+abs(rcut_tol), abs(rcut_tol))
    else
        return 1.
    end
end

"""
    get_rllm(overlaps, conf=get_empty_config(); load_rllm=get_load_rllm(conf), rllm_file=get_rllm_file(conf), interpolate_rllm=get_interpolate_rllm(conf))

Retrieves or computes the radial orbital integral look-up table (RLLM) for a given set of overlaps, based on configuration settings.

# Arguments
- `overlaps::Vector`: A list of overlap objects for which RLLM data is required.
- `conf::Config`: Configuration object controlling the behavior of RLLM retrieval or generation. Defaults to an empty configuration.
- `comm`: (Keyword argument) MPI communicator, defaults to `nothing`.
- `load_rllm::Bool`: (Keyword argument) If `true`, load the RLLM data from a file. The file location is provided by `rllm_file`. Defaults to the value from `conf`.
- `rllm_file::String`: (Keyword argument) Filename for loading or saving the RLLM data. Defaults to the value from `conf`.
- `interpolate_rllm::Bool`: (Keyword argument) If `true`, interpolate new RLLM data based on the overlaps and save it to a file. Defaults to the value from `conf`.
- `verbosity::Int`: (Keyword argument) Controls level of verbosity.

# Returns
- `rllm_dict::Dict{String, CubicSpline}`: A dictionary mapping overlap string representations to cubic spline interpolations of the radial integrals. If `load_rllm` is `true`, the data is read from the file. If `interpolate_rllm` is `true`, it is interpolated and saved.
"""
function get_rllm(overlaps, conf=get_empty_config();
                    comm=nothing,
                    load_rllm=get_load_rllm(conf), 
                    rllm_file=get_rllm_file(conf), 
                    interpolate_rllm=get_interpolate_rllm(conf), 
                    verbosity=get_verbosity(conf))
    
    rllm_dict = Dict{String, CubicSpline{Float64}}()
    if load_rllm
        if verbosity > 0; println(" Reading distance dependence from file..."); end
        time = @elapsed read_rllm(overlaps, comm, rllm_dict, filename=rllm_file)
        if verbosity > 0; println(" Finished in $time s."); end
    elseif interpolate_rllm
        i = 0
        Nover = length(overlaps)
        if verbosity > 0; println(" Interpolating distance dependence..."); end
        time = @elapsed Threads.@threads for overlap in overlaps
            rllm_dict[string(overlap, apply_oc=true)] = interpolate_overlap(overlap, conf)
            i += 1
            if verbosity > 0; println("   ($i / $Nover) ", string(overlap, apply_oc=true)); end
        end
        if verbosity > 0; println(" Finished in $time s."); end
        save_rllm(rllm_dict, comm, filename=rllm_file)
    end
    return rllm_dict
end

"""
    interpolate_overlap!(overlap, conf::Config)

Interpolates an overlap and stores the resulting cubic spline in the overlap's `rllm` field.

# Arguments
- `overlap`: An overlap object that contains information about the interaction type and orbital configuration.
- `conf::Config`: A configuration object that provides values for parameters such as `n` and `α` for each ion type involved in the overlap.

"""
function interpolate_overlap(overlap, conf::Config)
    itper = AdaptiveInterpolator(conf)
    ns = Int64[get_n(conf, number_to_element(overlap.ion_label.types[i])) for i in 1:2]
    αs = Float64[get_alpha(conf, number_to_element(overlap.ion_label.types[i])) for i in 1:2]
    f(r) = distance_dependence(overlap.type, overlap.orbconfig, r, ns, αs)
    xs, ys = interpolate_f(itper, f)
    return CubicSpline(xs, ys)
end

"""
    distance_dependence(ME::MatrixElement, orbconfig::OrbitalConfiguration, r, ns, αs)

Computes the distance-dependent interaction between two orbitals, based on their matrix element `ME` and orbital configuration `orbconfig`.

# Arguments
- `ME::MatrixElement`: The matrix element representing the interaction type.
- `orbconfig::OrbitalConfiguration`: The configuration of the orbitals involved in the interaction, which determines the orbital angular momenta.
- `r`: The distance between the two orbitals in the interaction.
- `ns`: A vector containing the quantum numbers `n` for the two orbitals.
- `αs`: A vector containing the exponential decay factors `α` for the two orbitals.

# Returns
- The integral `I` of the overlap function between the two orbitals over the given distance `r`, modified by a sign factor if necessary.
"""
function distance_dependence(ME, orbconfig, r, ns, αs)::Float64
    R₁ = ndict[ns[1]](αs[1]); R₂ = ndict[ns[2]](αs[2])
    xmin = [-10, -10, -10]; xmax = [15+r/3, 15+r/3, r+15]
    param = string(ME)
    m = mdict_inv[param[3]]
    l₁, l₂ = typeof(orbconfig) == MirrOrb ? (ldict_inv[param[2]], ldict_inv[param[1]]) : (ldict_inv[param[1]], ldict_inv[param[2]])
    sign = l₁ > l₂ ? (-1)^(l₁+l₂) : 1
    Y₁, Y₂ = get_spherical.([l₁, l₂], m)
    f = Overlap(Y₁, R₁, zeros(3), Y₂, R₂, [0., 0., r])
    I, _ = hcubature(f, xmin, xmax, rtol=1e-5, maxevals=1000000, initdiv=5)
    return sign*I
end

"""
    save_rllm(overlaps; filename="rllm.dat")

Saves overlap data, including the associated Rllm values, to a file.

# Arguments
- `rllm_dict`: A collection of overlap objects, where each overlap contains an `rllm` field with `xs` (x values) and `ys` (y values).
- `comm`: The MPI communicator.
- `filename::String`: The name of the file where the Rllm data will be saved. Defaults to `"rllm.dat"`.
"""
function save_rllm(rllm_dict, comm; filename="rllm.dat")
    if occursin(".h5", filename)
        if isnothing(comm)
            h5open(filename, "cw") do file
                for (overlap, spline) in rllm_dict
                    if !haskey(file, overlap)
                        file[overlap] = hcat(spline.xs, spline.ys)
                    end
                end
            end
        else
            h5open(filename, "cw", comm) do file
                for (overlap, spline) in rllm_dict
                    if !haskey(file, overlap)
                        file[overlap] = hcat(spline.xs, spline.ys)
                    end
                end
            end
        end
    else
        file = open(filename, "w")
        for (overlap, spline) in rllm_dict
            println(file, overlap)
            for x in spline.xs
                print(file, "  "*string(x))
            end
            print(file, "\n")
            for y in spline.ys
                print(file, "  "*string(y))
            end
            print(file, "\n")
        end
        close(file)
    end
end

"""
    read_rllm(filename="rllm.dat") -> Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}

Reads the `rllm.dat` file and returns a dictionary mapping overlap labels to tuples of vectors containing the `x` and `y` values.

# Arguments
- `filename::String`: The name of the file containing Rllm data. Defaults to `"rllm.dat"`.

# Returns
- A dictionary where each key is an overlap label, and each value a tuple of x/y value vectors.
"""
function read_rllm(overlaps, comm, rllm_dict=Dict{String, CubicSpline{Float64}}(); filename="rllm.dat")
    if occursin(".h5", filename)
        if isnothing(comm)
            h5open(filename, "r") do file
                for overlap in overlaps
                    overlap_str = string(overlap, apply_oc=true)
                    data = file[overlap_str]
                    rllm_dict[overlap_str] = CubicSpline(data[:, 1], data[:, 2])
                end
            end
        else
            h5open(filename, "r", comm) do file
                for overlap in overlaps
                    overlap_str = string(overlap, apply_oc=true)
                    data = file[overlap_str]
                    rllm_dict[overlap_str] = CubicSpline(data[:, 1], data[:, 2])
                end
            end
        end
    else
        lines = open_and_read(filename)
        lines = split_lines(lines)
        for i in 1:3:length(lines)
            overlap = lines[i][1]
            xs = parse.(Float64, lines[i+1])
            ys = parse.(Float64, lines[i+2])
            rllm_dict[overlap] = CubicSpline(xs, ys)
        end
        return rllm_dict
    end
end