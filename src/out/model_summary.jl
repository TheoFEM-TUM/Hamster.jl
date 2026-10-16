"""
    correction_statistics(Δε)

Summary statistics of a set of onsite corrections `Δε`: number of sites, width `σ`, extrema, 1% and 99% percentiles,
the tail ratio `max|Δε| / σ` and the fraction of sites with `|Δε| > 3σ`.
"""
function correction_statistics(Δε::AbstractVector)
    isempty(Δε) && return (n=0, σ=NaN, min=NaN, p1=NaN, p99=NaN, max=NaN, tail=NaN, frac3σ=NaN)
    σ = std(Δε, corrected=false)
    return (n=length(Δε), σ=σ, min=minimum(Δε), p1=quantile(Δε, 0.01), p99=quantile(Δε, 0.99), max=maximum(Δε),
            tail=σ > 0 ? maximum(abs, Δε) / σ : 0.0, frac3σ=σ > 0 ? count(x -> abs(x) > 3σ, Δε) / length(Δε) : 0.0)
end

"""
    write_model_summary(ewald::EwaldOnsites, comm, conf; kwargs...)

Append a summary of the Ewald onsite model to `hamster.out` (only if the file exists, rank 0 writes).

For every species, the onsite corrections `Δε = -γ (φ - φ̄)` of all sites of all structures (gathered over the MPI
ranks) are summarized: subtracted species mean `φ̄`, width, extrema, percentiles, tail ratio and the spread of the
per-structure species means. The latter is the part of the correction that is removed if the model is built for a
single structure, since the species means are taken over the structures of the current run.
"""
function write_model_summary(ewald::EwaldOnsites, comm, conf=get_empty_config();
        rank=0,
        label="",
        species_mean,
        structure_means,
        charge_types,
        method=get_ewald_method(conf),
        rcut=get_ewald_rcut(conf),
        alpha=get_ewald_alpha(conf),
        mesh_spacing=get_ewald_mesh_spacing(conf),
        filename="hamster.out")

    nspecies = length(ewald.param_labels)
    local_Δε = [Float64[] for _ in 1:nspecies]
    for n in eachindex(ewald.potentials), (ϕ, type) in zip(ewald.potentials[n], ewald.types_per_strc[n])
        p = findfirst(==(type), ewald.param_labels)
        push!(local_Δε[p], -1000 * ewald.params[p] * ϕ)
    end

    all_Δε = MPI.gather(local_Δε, comm, root=0)
    all_structure_means = MPI.gather(structure_means, comm, root=0)
    all_charge_types = MPI.gather(charge_types, comm, root=0)
    nstrc = MPI.Reduce(length(ewald.potentials), +, comm, root=0)
    (rank ≠ 0 || !isfile(filename)) && return nothing

    Δε = [reduce(vcat, (Δε_rank[p] for Δε_rank in all_Δε), init=Float64[]) for p in 1:nspecies]
    structure_means = reduce(vcat, all_structure_means)
    charge_types = sort(unique(reduce(vcat, all_charge_types)))
    α = alpha == 0 ? 4.0 / rcut : alpha

    open(filename, "a") do io
        write_separator(io)
        println(io, "  Ewald Model Summary" * (isempty(label) ? "" : " ($label)"))
        write_separator(io, char='-')
        @printf(io, "      %-28s %s\n", "method:", method)
        @printf(io, "      %-28s %.4f 1/Å%s\n", "alpha:", α, alpha == 0 ? " (4 / rcut)" : "")
        @printf(io, "      %-28s %.2f Å\n", "rcut:", rcut)
        method == "ewald" && @printf(io, "      %-28s %.2f Å\n", "mesh_spacing:", mesh_spacing)
        @printf(io, "      %-28s %d\n", "structures:", nstrc)
        @printf(io, "      %-28s %s\n", "update:", ewald.update)
        @printf(io, "      %-28s %s\n", "qeff:", join(["$(number_to_element(t)) $(get_qeff(conf, number_to_element(t)))" for t in charge_types], ", "))
        if method == "ewald" && rcut > get_grid_size(conf)
            @printf(io, "      WARNING: Ewald rcut (%.2f Å) > grid_size (%.2f Å): real-space pairs beyond the\n", rcut, get_grid_size(conf))
            println(io, "               neighbor grid are dropped; increase grid_size or reduce the Ewald rcut.")
        end
        println(io)
        println(io, "  Onsite corrections Δε = -γ (φ - φ̄) per species (energies in meV):")
        @printf(io, "    %-7s %7s %7s %6s %9s %7s %8s %8s %8s %8s %6s %6s %7s\n",
                "species", "sites", "qeff", "γ", "φ̄ [V]", "σ", "min", "p1", "p99", "max", "max/σ", ">3σ %", "σ_strc")
        for (p, type) in enumerate(ewald.param_labels)
            s = correction_statistics(Δε[p])
            means = [m for (t, m) in structure_means if t == type]
            σ_strc = length(means) > 1 ? 1000 * ewald.params[p] * std(means, corrected=false) : NaN
            element = number_to_element(type)
            @printf(io, "    %-7s %7d %7.3f %6.3f %9.4f %7.1f %8.1f %8.1f %8.1f %8.1f %6.2f %6.2f %7s\n",
                    element, s.n, get_qeff(conf, element), ewald.params[p], species_mean[p], s.σ, s.min, s.p1, s.p99, s.max,
                    s.tail, 100 * s.frac3σ, isnan(σ_strc) ? "-" : @sprintf("%.1f", σ_strc))
        end
        println(io)
        println(io, "    φ̄: species mean of the site potentials over all structures of this run (subtracted).")
        println(io, "    σ: width of Δε; max/σ: largest |Δε| relative to σ; >3σ %: fraction of sites with |Δε| > 3σ.")
        println(io, "    σ_strc: spread of the per-structure species means, i.e. the part of Δε that is lost if the")
        println(io, "            model is built for a single structure.")
        write_separator(io)
    end
    return nothing
end
