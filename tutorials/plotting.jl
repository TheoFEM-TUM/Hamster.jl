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
    L_val = all(iszero, L_val) ? Float64[] : L_val                                       # fit without validation
    if HAS_PLOTS
        L = vcat(L_train, L_val)
        p = Plots.plot(L_train, xlabel="iteration", ylabel="loss (eV)", label="training", lw=2,
                       yscale=maximum(L) > 10minimum(L) ? :log10 : :identity,                 # log scale for large ranges
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

"""
    plot_dos(file="dos.h5"; structure=nothing, out="dos.png", xlims=nothing)

DOS and PDOS (left) and -COHP (right) of a DOS post-processing run (first structure by default), energies relative
to the Fermi energy if it was computed (`nelec`).
"""
function plot_dos(file="dos.h5"; structure=nothing, out="dos.png", xlims=nothing)
    h5open(file) do f
        g = f[isnothing(structure) ? first(keys(f)) : string(structure)]
        E = read(g["energies"])
        Ef = haskey(g, "fermi_energy") ? read(g["fermi_energy"]) : 0.0
        pdos = haskey(g, "pdos") ? Dict(k => read(g["pdos"][k]) for k in keys(g["pdos"])) : Dict{String, Vector{Float64}}()
        dos = haskey(g, "dos") ? read(g["dos"]) : sum(values(pdos))                       # the PDOS add up to the DOS
        cohp = haskey(g, "cohp") ? Dict(k => read(g["cohp"][k]) for k in keys(g["cohp"])) : Dict{String, Vector{Float64}}()
        haskey(g, "fermi_energy") && @printf("Fermi energy: %.3f eV (energies below are relative to it)\n", Ef)
        for k in sort(collect(keys(cohp)))
            haskey(g, "icohp_fermi") && haskey(g["icohp_fermi"], k) && @printf("ICOHP %-6s: %10.2f eV\n", k, read(g["icohp_fermi"][k]))
        end
        if HAS_PLOTS
            xl = isnothing(xlims) ? :auto : xlims
            p1 = Plots.plot(E .- Ef, dos, label="total", color=:black, lw=1.5, xlabel="E - E_F (eV)", ylabel="states / eV",
                            framestyle=:box, grid=false, xlims=xl)
            for k in sort(collect(keys(pdos)))
                Plots.plot!(p1, E .- Ef, pdos[k], label=k, lw=1.5)
            end
            p2 = Plots.plot(xlabel="E - E_F (eV)", ylabel="-COHP", framestyle=:box, grid=false, xlims=xl)
            Plots.hline!(p2, [0], color=:gray, label="")
            for k in sort(collect(keys(cohp)))
                Plots.plot!(p2, E .- Ef, -cohp[k], label=k, lw=1.5)
            end
            Plots.vline!(p1, [0], color=:gray, ls=:dash, label=""); Plots.vline!(p2, [0], color=:gray, ls=:dash, label="")
            Plots.savefig(Plots.plot(p1, p2, layout=(1, 2), size=(1000, 400), margin=5Plots.mm), out)
            println("Figure written to ", out)
        else
            csv = replace(out, r"\.\w+$" => ".csv")
            labels = vcat(sort(collect(keys(pdos))), "cohp " .* sort(collect(keys(cohp))))
            cols = hcat(E .- Ef, dos, [pdos[k] for k in sort(collect(keys(pdos)))]..., [cohp[k] for k in sort(collect(keys(cohp)))]...)
            writedlm(csv, cols, ','); println("DOS written to ", csv, " (columns: E - E_F, DOS, ", join(labels, ", "), ")")
        end
    end
end

"""
    band_edges_sparse(file="Es.dat"; eig_target) -> (vbm, cbm)

Band edges from a sparse eigensolver run (only the eigenvalues closest to `eig_target`): the highest eigenvalue below
and the lowest above `eig_target`, which must lie in the band gap.
"""
function band_edges_sparse(file="Es.dat"; eig_target)
    E = vec(read_bands(file))
    vbm, cbm = maximum(filter(<(eig_target), E)), minimum(filter(>(eig_target), E))
    @printf("Highest state below eig_target: %.3f eV, lowest above: %.3f eV, gap: %.3f eV\n", vbm, cbm, cbm - vbm)
    return vbm, cbm
end

"""
    plot_hyperopt(file="hamster_out.h5"; x, y, out="hyperopt.png")

Validation loss of every trial of a hyperparameter optimization over two hyperparameters `x` and `y` (names as in the
`params` tag, e.g. `"Ga_alpha"`), and the best trial. The score of a trial is its lowest validation loss (or training
loss, without validation data).
"""
function plot_hyperopt(file="hamster_out.h5"; x, y, out="hyperopt.png")
    X, Y = h5read(file, x), h5read(file, y)
    L = vec(h5read(file, "L_train"))      # score of every trial: its validation loss (training loss without val_data)
    best = argmin(L)
    @printf("Best of %d trials: %s = %.3f, %s = %.3f (loss %.4f eV)\n", length(L), x, X[best], y, Y[best], L[best])
    if HAS_PLOTS
        p = Plots.scatter(X, Y, marker_z=L, color=:viridis, ms=7, label="trials", xlabel=x, ylabel=y, framestyle=:box,
                          grid=false, size=(600, 450), colorbar_title="loss (eV)")
        Plots.scatter!(p, [X[best]], [Y[best]], marker=:star5, ms=14, color=:red, label="best")
        Plots.savefig(p, out); println("Figure written to ", out)
    else
        csv = replace(out, r"\.\w+$" => ".csv")
        writedlm(csv, hcat(X, Y, L), ','); println("Trials written to ", csv, " (columns: $x, $y, loss)")
    end
    return X[best], Y[best]
end
