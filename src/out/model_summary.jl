summary_title(model_name, label) = "  $model_name Model Summary" * (isempty(label) ? "" : " ($label)")
range_string(a, b) = a == b ? string(a) : "$a - $b"

const SUMMARY_WIDTH = 80

function write_summary_line(io, key, value)
    prefix = rpad("      $key:", 34) * " "
    lines = wrap_text(string(value), SUMMARY_WIDTH - textwidth(prefix))
    isempty(lines) && (lines = [""])
    println(io, prefix, lines[1])
    foreach(line -> println(io, " "^textwidth(prefix), line), lines[2:end])
end

function write_table_row(io, cells, widths)
    padded = [i == 1 ? rpad(cell, width) : lpad(cell, width) for (i, (cell, width)) in enumerate(zip(cells, widths))]
    println(io, "    ", join(padded, " "))
end

function write_legend(io, text; indent=4)
    lines = wrap_text(text, SUMMARY_WIDTH - indent - 2)
    isempty(lines) && return nothing
    println(io, " "^indent, lines[1])
    foreach(line -> println(io, " "^(indent + 2), line), lines[2:end])
end

format_number(x, digits) = isnan(x) ? "-" : Printf.format(Printf.Format("%.$(digits)f"), x)

"""
    correction_statistics(Δε)

Summary statistics of the onsite corrections `Δε`.
"""
function correction_statistics(Δε::AbstractVector)
    isempty(Δε) && return (n=0, σ=NaN, min=NaN, p1=NaN, p99=NaN, max=NaN, tail=NaN, frac3σ=NaN)
    σ = std(Δε, corrected=false)
    return (n=length(Δε), σ=σ, min=minimum(Δε), p1=quantile(Δε, 0.01), p99=quantile(Δε, 0.99), max=maximum(Δε),
            tail=σ > 0 ? maximum(abs, Δε) / σ : 0.0, frac3σ=σ > 0 ? count(x -> abs(x) > 3σ, Δε) / length(Δε) : 0.0)
end

"""
    write_model_summary(ewald::EwaldOnsites, comm, conf; kwargs...)

Append a summary of the Ewald onsite model to `hamster.out`.
"""
function write_model_summary(ewald::EwaldOnsites, comm, conf=get_empty_config();
        rank=0,
        label="",
        species_mean,
        structure_means,
        rcut=get_ewald_rcut(conf),
        alpha=get_ewald_alpha(conf),
        grid_rebuilt=false,
        filename="hamster.out")

    nspecies = length(ewald.param_labels)
    local_Δε = [Float64[] for _ in 1:nspecies]
    for n in eachindex(ewald.potentials), (ϕ, type) in zip(ewald.potentials[n], ewald.types_per_strc[n])
        p = findfirst(==(type), ewald.param_labels)
        push!(local_Δε[p], -1000 * ewald.params[p] * ϕ)
    end

    all_Δε = MPI.gather(local_Δε, comm, root=0)
    all_structure_means = MPI.gather(structure_means, comm, root=0)
    nstrc = MPI.Reduce(length(ewald.potentials), +, comm, root=0)
    (rank ≠ 0 || !isfile(filename)) && return nothing

    Δε = [reduce(vcat, (Δε_rank[p] for Δε_rank in all_Δε), init=Float64[]) for p in 1:nspecies]
    structure_means = reduce(vcat, all_structure_means)

    open(filename, "a") do io
        println(io)
        write_separator(io)
        println(io, summary_title("Ewald", label))
        write_separator(io, char='-')
        elements = number_to_element.(ewald.param_labels)
        write_summary_line(io, "structures", nstrc)
        alpha == 0 && write_summary_line(io, "alpha (4 / rcut)", @sprintf("%.4f 1/Å", 4.0 / rcut))
        grid_rebuilt && write_summary_line(io, "real-space neighbor grid", "rebuilt (Ewald rcut > TB rcut or grid_size)")
        write_summary_line(io, "subtracted species mean φ̄", join(["$e $(@sprintf("%.4f", m)) V" for (e, m) in zip(elements, species_mean)], ", "))
        println(io)
        println(io, "  Onsite corrections Δε = -γ (φ - φ̄) per species (meV):")
        widths = [7, 6, 6, 7, 7, 7, 7, 6, 6, 6]
        write_table_row(io, ["species", "sites", "σ", "min", "p1", "p99", "max", "max/σ", ">3σ %", "σ_strc"], widths)
        for (p, type) in enumerate(ewald.param_labels)
            s = correction_statistics(Δε[p])
            means = [m for (t, m) in structure_means if t == type]
            σ_strc = length(means) > 1 ? 1000 * ewald.params[p] * std(means, corrected=false) : NaN
            write_table_row(io, [elements[p], string(s.n), format_number(s.σ, 1), format_number(s.min, 1), format_number(s.p1, 1),
                                 format_number(s.p99, 1), format_number(s.max, 1), format_number(s.tail, 2),
                                 format_number(100 * s.frac3σ, 2), format_number(σ_strc, 1)], widths)
        end
        println(io)
        write_legend(io, "φ̄: species mean of the site potentials over all structures of this run.")
        write_legend(io, "sites: number of sites of the species in all structures.")
        write_legend(io, "σ: standard deviation of Δε; min, max: smallest and largest Δε.")
        write_legend(io, "p1, p99: 1st and 99th percentiles, i.e. 1% of the sites have a Δε below p1 and 1% above p99.")
        write_legend(io, "max/σ: largest |Δε| relative to σ; large values indicate a heavy tail.")
        write_legend(io, ">3σ %: percentage of sites with |Δε| > 3σ (0.27% for a Gaussian distribution).")
        write_legend(io, "σ_strc: spread of the per-structure species means, i.e. the part of Δε that is lost if the " *
                         "model is built for a single structure.")
        write_separator(io)
    end
    return nothing
end

"""
    write_model_summary(model::TBModel, strcs, bases, comm; kwargs...)

Append a summary of the TB model to `hamster.out`.
"""
function write_model_summary(model::TBModel, strcs, bases, comm; rank=0, label="", filename="hamster.out")
    natoms = isempty(strcs) ? [0] : [length(strc.ions) for strc in strcs]
    norbs = isempty(bases) ? [0] : [length(basis) for basis in bases]
    nRs = isempty(strcs) ? [0] : [size(strc.Rs, 2) for strc in strcs]
    entries = sum((nnz(hvR) for h in model.hs for hvR in h), init=0)

    nstrc = MPI.Reduce(length(strcs), +, comm, root=0)
    natoms_min, natoms_max = MPI.Reduce(minimum(natoms), min, comm, root=0), MPI.Reduce(maximum(natoms), max, comm, root=0)
    norbs_min, norbs_max = MPI.Reduce(minimum(norbs), min, comm, root=0), MPI.Reduce(maximum(norbs), max, comm, root=0)
    norbs_total = MPI.Reduce(sum(norbs), +, comm, root=0)
    nRs_min, nRs_max = MPI.Reduce(minimum(nRs), min, comm, root=0), MPI.Reduce(maximum(nRs), max, comm, root=0)
    entries = MPI.Reduce(entries, +, comm, root=0)
    (rank ≠ 0 || !isfile(filename)) && return nothing

    shells = sort(unique(Int64(p.nnlabel) for p in model.param_labels))
    shell_counts = ["$(shell == 0 ? "onsite" : "NN$shell") $(count(p -> p.nnlabel == shell, model.param_labels))" for shell in shells]
    pairs = sort(unique(string(p.ion_label) for p in model.param_labels))
    pair_counts = ["$pair $(count(p -> string(p.ion_label) == pair, model.param_labels))" for pair in pairs]

    open(filename, "a") do io
        println(io)
        write_separator(io)
        println(io, summary_title("TB", label))
        write_separator(io, char='-')
        write_summary_line(io, "structures", nstrc)
        write_summary_line(io, "atoms per structure", range_string(natoms_min, natoms_max))
        write_summary_line(io, "orbitals per structure", range_string(norbs_min, norbs_max) * " (total $norbs_total)")
        write_summary_line(io, "R vectors per structure", range_string(nRs_min, nRs_max))
        write_summary_line(io, "parameters", length(model.param_labels))
        write_summary_line(io, "  per shell", join(shell_counts, ", "))
        write_summary_line(io, "  per ion pair", join(pair_counts, ", "))
        write_summary_line(io, "geometry tensor entries", @sprintf("%d (%.1f MB)", entries, entries * 16 / 1024^2))
        write_separator(io)
    end
    return nothing
end

"""
    descriptor_groups(strc, basis)

Ion pair (e.g. `"Br+Pb"`) or onsite label (e.g. `"Pb onsite"`) of every matrix element of the ML descriptor.
"""
function descriptor_groups(strc, basis)
    ion_of_orbital = [iion for iion in eachindex(basis.orbitals) for _ in basis.orbitals[iion]]
    elements = number_to_element.(get_ion_types(strc.ions))
    R0 = findR0(strc.Rs)
    return (i, j, R) -> begin
        a, b = ion_of_orbital[i], ion_of_orbital[j]
        a == b && R == R0 ? "$(elements[a]) onsite" : join(sort([elements[a], elements[b]]), "+")
    end
end

"""
    max_similarities(descriptors, data_points, σ)

Similarity of every descriptor to its most similar data point, `max_k exp(-|x - d_k|² / 2σ²)`.
"""
function max_similarities(descriptors, data_points, σ)
    isempty(data_points) && return zeros(length(descriptors))
    return tmap(descriptors) do x
        d2 = minimum(d -> sum(abs2, x - d), data_points)
        exp(-d2 / (2σ^2))
    end
end

const NBINS_COVERAGE = 1000

function histogram_quantile(counts, q)
    target = q * sum(counts)
    cumulative = 0
    for (b, c) in enumerate(counts)
        cumulative += c
        cumulative ≥ target && c > 0 && return b / length(counts)
    end
    return 1.0
end

"""
    write_model_summary(kernel::HamiltonianKernel, strcs, bases, comm, conf; kwargs...)

Append a summary of the ML model to `hamster.out`.
"""
function write_model_summary(kernel::HamiltonianKernel, strcs, bases, comm, conf=get_empty_config();
        rank=0, label="", sim_tol=get_ml_sim_tol(conf), filename="hamster.out")

    histograms = Dict{String, Vector{Int64}}()
    minima = Dict{String, Float64}()
    for n in eachindex(strcs)
        group = descriptor_groups(strcs[n], bases[n])
        for R in eachindex(kernel.structure_descriptors[n])
            is, js, xs = findnz(kernel.structure_descriptors[n][R])
            sims = max_similarities(xs, kernel.data_points, kernel.sim_params)
            for (i, j, s) in zip(is, js, sims)
                g = group(i, j, R)
                histogram = get!(() -> zeros(Int64, NBINS_COVERAGE), histograms, g)
                histogram[clamp(ceil(Int64, s * NBINS_COVERAGE), 1, NBINS_COVERAGE)] += 1
                minima[g] = min(get(minima, g, 1.0), s)
            end
        end
    end

    all_histograms = MPI.gather(histograms, comm, root=0)
    all_minima = MPI.gather(minima, comm, root=0)
    (rank ≠ 0 || !isfile(filename)) && return nothing

    histograms = mergewith(+, all_histograms...)
    minima = mergewith(min, all_minima...)
    groups = sort(collect(keys(histograms)), by=g -> (!endswith(g, "onsite"), g))
    total = sum(sum, values(histograms), init=0)

    open(filename, "a") do io
        println(io)
        write_separator(io)
        println(io, summary_title("ML", label))
        write_separator(io, char='-')
        write_summary_line(io, "descriptor vectors", total)
        println(io)
        println(io, "  Coverage: similarity of every descriptor to its most similar data point")
        widths = [12, 11, 6, 6, 6, 6, 7, 7]
        write_table_row(io, ["group", "descriptors", "min", "p1", "p5", "median", "<tol %", "<0.5 %"], widths)
        for g in groups
            h = histograms[g]
            below_tol = sum(h[1:clamp(floor(Int64, sim_tol * NBINS_COVERAGE), 0, NBINS_COVERAGE)], init=0)
            below_half = sum(h[1:NBINS_COVERAGE ÷ 2])
            write_table_row(io, [g, string(sum(h)), @sprintf("%.3f", minima[g]), @sprintf("%.3f", histogram_quantile(h, 0.01)),
                                 @sprintf("%.3f", histogram_quantile(h, 0.05)), @sprintf("%.3f", histogram_quantile(h, 0.5)),
                                 @sprintf("%.2f", 100 * below_tol / sum(h)), @sprintf("%.2f", 100 * below_half / sum(h))], widths)
        end
        println(io)
        write_legend(io, "Similarity exp(-|x - d|² / 2σ²) with σ = sim_params: 1 for a descriptor x equal to a data " *
                         "point d, close to 0 far from all data points.")
        write_legend(io, "group: ion pair of the matrix elements, or onsite terms of an element.")
        write_legend(io, "descriptors: number of descriptor vectors (matrix elements) of the group.")
        write_legend(io, "min: smallest similarity; p1, p5, median: 1st, 5th and 50th percentiles, i.e. 1%, 5% and " *
                         "50% of the descriptors have a smaller similarity (resolution 0.001).")
        write_legend(io, "<tol %: percentage of descriptors with a similarity below sim_tol; <0.5 %: below 0.5.")
        write_legend(io, "Descriptors with a small similarity to all data points are outside the training data; " *
                         "there, the ML correction is close to zero and the model falls back to TB.")
        write_separator(io)
    end
    return nothing
end
