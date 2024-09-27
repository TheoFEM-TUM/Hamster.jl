function get_rllm(overlaps, conf=get_empty_config(); load_rllm=get_load_rllm(conf), rllm_file=get_rllm_file(conf), interpolate_rllm=get_interpolate_rllm(conf))
    if load_rllm
        rllm_dict = read_rllm(rllm_file)
        return rllm_dict
    elseif interpolate_rllm
        rllm_dict = Dict{String, CubicSpline{Float64}}()
        i = 0
        Nover = length(overlaps)
        Threads.@threads for overlap in overlaps
            rllm_dict[string(overlap, apply_oc=true)] = interpolate_overlap(overlap, conf)
            i += 1
            println("   ($i / $Nover)")
        end
        save_rllm(rllm_dict, filename=rllm_file)
        return rllm_dict
    end
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
- `overlaps`: A collection of overlap objects, where each overlap contains an `rllm` field with `xs` (x values) and `ys` (y values).
- `filename::String`: The name of the file where the Rllm data will be saved. Defaults to `"rllm.dat"`.
"""
function save_rllm(rllm_dict; filename="rllm.dat")
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

"""
    read_rllm(filename="rllm.dat") -> Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}

Reads the `rllm.dat` file and returns a dictionary mapping overlap labels to tuples of vectors containing the `x` and `y` values.

# Arguments
- `filename::String`: The name of the file containing Rllm data. Defaults to `"rllm.dat"`.

# Returns
- A dictionary where each key is an overlap label, and each value a tuple of x/y value vectors.
"""
function read_rllm(filename="rllm.dat")
    Rllm_dict = Dict{String, CubicSpline{Float64}}()
    lines = open_and_read(filename)
    lines = split_lines(lines)
    for i in 1:3:length(lines)
        overlap = lines[i][1]
        xs = parse.(Float64, lines[i+1])
        ys = parse.(Float64, lines[i+2])
        Rllm_dict[overlap] = CubicSpline(xs, ys)
    end
    return Rllm_dict
end