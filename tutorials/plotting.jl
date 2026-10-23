# Plotting helpers for the Hamster tutorials.
#
# They read Hamster's output files (Es.dat, hamster_out.h5) and the DFT data of the tutorials, and plot them with
# Plots.jl if it is available in your Julia environment. Plots.jl is not a dependency of Hamster (it is large); install
# it once in your default environment with
#
#     julia -e 'using Pkg; Pkg.add("Plots")'
#
# Without Plots.jl, the data of each figure is written to a CSV file that you can plot with any program.
#
# Usage in a tutorial folder: julia --project=<Hamster> plot.jl   (plot.jl includes this file)

using Hamster, HDF5, Printf, DelimitedFiles

const HAS_PLOTS = try
    @eval import Plots
    Plots.gr(); ENV["GKSwstype"] = "100"           # draw without a window (also on clusters)
    true
catch
    @info "Plots.jl is not available: writing CSV files instead of figures (julia -e 'using Pkg; Pkg.add(\"Plots\")')."
    false
end

const DATA = joinpath(@__DIR__, "data")

"""
    read_bands(file) -> Matrix (bands × k-points)

Eigenvalues of a Hamster run (`Es.dat`, first structure) or of a VASP EIGENVAL file.
"""
function read_bands(file)
    if occursin("EIGENVAL", basename(file))
        return Hamster.read_eigenval(file)[2]
    else
        lines = readlines(file)
        nb, nk = parse.(Int, split(lines[1])[1:2])
        return reshape(parse.(Float64, lines[2:1+nb*nk]), nb, nk)
    end
end

"""
    plot_bands(file; reference=nothing, nshift=0, title="", out="bands.png", ylims=nothing)

Band structure of a Hamster run (`Es.dat`) along the k-points of the run, optionally compared with DFT bands
(`reference`, an EIGENVAL file with the same k-points). `nshift`: number of DFT bands below the first model band.
"""
function plot_bands(file; reference=nothing, nshift=0, title="", out="bands.png", ylims=nothing)
    E = read_bands(file)
    ref = isnothing(reference) ? nothing : read_bands(reference)[nshift+1:nshift+size(E, 1), :]
    if HAS_PLOTS
        p = Plots.plot(xlabel="k-point", ylabel="energy (eV)", title=title, legend=:topright, framestyle=:box, grid=false,
                       size=(650, 450), ylims=isnothing(ylims) ? :auto : ylims)
        isnothing(ref) || Plots.plot!(p, transpose(ref), color=:black, lw=2.5, alpha=0.35, label=["DFT" fill("", 1, size(ref, 1) - 1)])
        Plots.plot!(p, transpose(E), color="#2a78d6", lw=1.5, label=["Hamster" fill("", 1, size(E, 1) - 1)])
        Plots.savefig(p, out); println("Figure written to ", out)
    else
        csv = replace(out, r"\.\w+$" => ".csv")
        writedlm(csv, isnothing(ref) ? transpose(E) : hcat(transpose(E), transpose(ref)), ',')
        println("Bands written to ", csv, " (columns: model bands", isnothing(ref) ? "" : ", then DFT bands", ")")
    end
    isnothing(ref) || @printf("Mean absolute deviation from DFT: %.3f eV\n", sum(abs, E .- ref) / length(E))
    return E
end

"""
    plot_losses(file="hamster_out.h5"; out="losses.png")

Training and validation loss of a fit against the iterations.
"""
function plot_losses(file="hamster_out.h5"; out="losses.png")
    L_train = vec(sum(h5read(file, "L_train"), dims=1)) ./ size(h5read(file, "L_train"), 1)   # mean over batches
    L_val = h5open(f -> haskey(f, "L_val") ? read(f["L_val"]) : Float64[], file)
    if HAS_PLOTS
        p = Plots.plot(L_train, xlabel="iteration", ylabel="loss (eV)", label="training", lw=2, yscale=:log10,
                       framestyle=:box, grid=false, size=(600, 400), color="#2a78d6")
        isempty(L_val) || Plots.plot!(p, L_val, label="validation", lw=2, ls=:dash, color="#eb6834")
        Plots.savefig(p, out); println("Figure written to ", out)
    else
        csv = replace(out, r"\.\w+$" => ".csv")
        writedlm(csv, isempty(L_val) ? L_train : hcat(L_train, L_val), ','); println("Losses written to ", csv)
    end
    @printf("Final training loss: %.4f eV", L_train[end]); isempty(L_val) || @printf(", validation loss: %.4f eV", L_val[end])
    println()
    return L_train, L_val
end
