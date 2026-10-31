"""
    EwaldOnsites{P,T,N}

Container for electrostatic onsite corrections computed from Ewald-type
methods (Ewald, Wolf, Zahn) over a set of structures.

This struct stores per-atom electrostatic potentials (mean-centered
per species) together with the metadata required to map them onto TB
onsite parameters.

# Fields
- `params::Vector{Float64}`:
    Trainable or fixed parameters associated with each atomic species (e.g.
    coupling strengths between electrostatic potential and onsite energies).

- `param_labels::Vector{UInt8}`:
    Unique list of atomic species identifiers corresponding to `params`.

- `potentials::Vector{P}`:
    Electrostatic potentials for each structure. Each entry is a vector of
    site potentials (one per atom), typically computed via an Ewald-like method
    and centered per species.

- `types_per_strc::Vector{T}`:
    Atomic species (as `UInt8`) for each atom in each structure.

- `norb_per_strc::Vector{N}`:
    Number of orbitals per structure (used to map site potentials to Hamiltonian
    orbital indices).

- `Rs_info::Matrix{Int64}`:
    Auxiliary information about lattice translations per structure:
    - row 1: number of R-vectors
    - row 2: index of the reference cell (R = 0)

- `update::Bool`:
    Flag indicating whether onsite corrections should be updated during
    optimization/training.

# Description
The `EwaldOnsites` object encapsulates the electrostatic contribution to
tight-binding onsite energies, computed from effective charges via Ewald
or related methods. Potentials are typically centered per species across
all structures to remove arbitrary reference offsets.

This allows a linear mapping of the form:

    Δεᵢ ≈ γ_{s(i)} ⋅ φᵢ

where `φᵢ` are the stored potentials and `γ` are the parameters in `params`.

# Constructor
    EwaldOnsites(strcs, bases, comm, conf; kwargs...)

Builds the container by:

1. Computing electrostatic potentials for each structure
2. Gathering unique species across MPI ranks
3. Initializing per-species parameters
4. Computing global per-species mean potentials (MPI-reduced)
5. Subtracting these means from all site potentials (`centering = global`), or the species means of each structure
   (`centering = structure`)

# Notes
- Potentials are only defined up to a constant; per-species centering ensures
  consistency with onsite fitting.
- The choice of electrostatic method (Ewald, Wolf, Zahn) affects the physical
  interpretation of the stored potentials.
"""
mutable struct EwaldOnsites{P, T, N}
    params :: Vector{Float64}
    param_labels :: Vector{UInt8}
    potentials :: Vector{P}
    types_per_strc :: Vector{T}
    norb_per_strc :: Vector{N}
    Rs_info :: Matrix{Int64}
    update :: Bool
end

function EwaldOnsites(strcs::Vector{Structure}, bases::Vector{<:Basis}, comm, conf=get_empty_config(); 
        rank=0,
        method          = get_ewald_method(conf),
        rcut            = get_ewald_rcut(conf),
        alpha           = get_ewald_alpha(conf),
        mesh_spacing    = get_ewald_mesh_spacing(conf),
        slab_axis       = get_ewald_slab_axis(conf),
        slab_scale      = get_ewald_slab_scale(conf),
        charge_compensation = get_ewald_charge_compensation(conf),
        centering       = conf("centering", "Ewald") == "default" ? (slab_axis > 0 ? "structure" : "global") : get_ewald_centering(conf),
        update          = get_ewald_update(conf),
        label           = "",
        write_summary   = true)

    types_per_strc = Vector{UInt8}[]
    norb_per_strc = Vector{Int64}[]
    potentials = Vector{Float64}[]
    Rs_info = zeros(Int64, 2, length(strcs))
    grid_rebuilt = false
    α = ewald_alpha(method, rcut, alpha)

    for (n, strc) in enumerate(strcs)
        pos = get_ion_positions(strc.ions, apply_distortion=true)

        ion_types = get_ion_types(strc.ions)
        charges = [get_qeff(conf, type) for type in number_to_element.(ion_types)]
        # compensation of the net charge (charge_compensation), from the reference positions, so that it is the same for
        # all distortions of a structure. The surface charges model a smooth electronic charge and enter only the
        # plane-averaged long range of the slab; the charges next to a defect (local) enter the sum as well.
        slab_charges = charges .+ compensation_charges(charge_compensation, get_ion_positions(strc.ions, apply_distortion=false),
                                                       ion_types, charges, strc.lattice; slab_axis=slab_axis)
        charge_compensation == "local" && (charges = slab_charges)

        reuse_grid = rcut ≤ min(strc.point_grid.grid_size, get_rcut(conf))
        grid_rebuilt |= !reuse_grid
        if method == "ewald"
            ewald = ewald_sum(pos, charges, strc.lattice, 
                                Rs              = reuse_grid ? strc.Rs : nothing, 
                                point_grid      = reuse_grid ? strc.point_grid : nothing, 
                                rcut            = rcut,
                                mesh_spacing    = mesh_spacing,
                                method          = method, 
                                alpha           = α
                            )
        else
            ewald = ewald_sum(pos, charges, strc.lattice, rcut=rcut, method=method, alpha=α)
        end
        φ = ewald.potentials
        if slab_axis > 0
            # long range of a slab: plane-averaged part (wolf, zahn) or dipole correction (ewald), see ewald_slab.jl
            φ = φ .+ slab_scale .* slab_potentials(pos, slab_charges, strc.lattice, slab_axis; method=method, alpha=α)
        end

        ion_indices_with_orbs = collect(1:length(ion_types))
        ion_types_with_orbs = get_ion_types(strc.ions, conf, uniq=true, withorbitals=true)
        filter!(ind -> ion_types[ind] ∈ ion_types_with_orbs, ion_indices_with_orbs)

        push!(types_per_strc, ion_types[ion_indices_with_orbs])
        push!(norb_per_strc, size(bases[n])[ion_indices_with_orbs])
        push!(potentials, φ[ion_indices_with_orbs])
        Rs_info[1, n] = size(strc.Rs, 2)
        Rs_info[2, n] = findR0(strc.Rs)
    end

    param_labels_local = Iterators.flatten(types_per_strc)
    param_labels = MPI.gather(param_labels_local, comm, root=0)
    if rank == 0
        param_labels = unique(Iterators.flatten(param_labels))
        nparams = length(param_labels)
    else
        param_labels = UInt8[]
        nparams = 0
    end
    nparams = MPI.Bcast(nparams, 0, comm)

    if rank ≠ 0
        resize!(param_labels, nparams)
    end
    MPI.Bcast!(param_labels, comm, root=0)
    MPI.Barrier(comm)

    params = init_ewald_params(param_labels, conf)
    
    type_to_idx = Dict(t => i for (i, t) in enumerate(param_labels))

    local_sum = zeros(Float64, length(param_labels))
    local_count = zeros(Int64, length(param_labels))
    for n in eachindex(potentials)
        for (pot, type) in zip(potentials[n], types_per_strc[n])
            p = type_to_idx[type]
            local_sum[p] += pot
            local_count[p] += 1
        end
    end

    species_mean = similar(local_sum)
    for t in eachindex(local_sum)
        species_sum = MPI.Reduce(local_sum[t], +, comm)
        species_count = MPI.Reduce(local_count[t], +, comm)
        if rank == 0
            species_mean[t] = species_sum / species_count
        end
    end
    MPI.Bcast!(species_mean, comm, root=0)

    centering ∈ ("global", "structure") || error("centering = $centering: unknown, use global or structure.")
    structure_means = Tuple{UInt8, Float64}[]
    for n in eachindex(potentials)
        means = Dict(type => mean(potentials[n][types_per_strc[n] .== type]) for type in unique(types_per_strc[n]))
        append!(structure_means, [(type, m) for (type, m) in means])
        for i in eachindex(potentials[n])
            typ = types_per_strc[n][i]
            potentials[n][i] -= centering == "structure" ? means[typ] : species_mean[type_to_idx[typ]]
        end
    end

    model = EwaldOnsites(params, param_labels, potentials, types_per_strc, norb_per_strc, Rs_info, update)
    write_summary && write_model_summary(model, comm, conf, rank=rank, label=label, species_mean=species_mean, structure_means=structure_means,
                                         method=method, rcut=rcut, alpha=α, alpha_auto=alpha == 0, grid_rebuilt=grid_rebuilt && method == "ewald",
                                         slab_axis=slab_axis, slab_scale=slab_scale, charge_compensation=charge_compensation,
                                         centering=centering)
    return model
end

"""
    ewald_alpha(method, rcut, alpha)

Damping α of the Ewald sum: `alpha` if set (≠ 0), else 4 / rcut for `ewald`, where α only splits the sum into a real-
and a reciprocal-space part, and 3 / rcut for `wolf` and `zahn`, where α sets the screening length (erfc(3) ≈ 2⋅10⁻⁵).
"""
ewald_alpha(method, rcut, alpha) = alpha ≠ 0 ? Float64(alpha) : (lowercase(method) ∈ ("wolf", "zahn") ? 3.0 : 4.0) / rcut

"""
    ewald_settings(conf)

Settings that define the Ewald potentials: method, rcut, the damping α actually used and the charges `qeff`, written
to the header of the parameter file of a fit (see `apply_ewald_settings!`).
"""
function ewald_settings(conf)
    method, rcut = get_ewald_method(conf), get_ewald_rcut(conf)
    qeff = sort([(type, get_qeff(conf, type)) for (type, block) in conf.blocks if haskey(block, "qeff")])
    return (method=method, rcut=rcut, alpha=ewald_alpha(method, rcut, get_ewald_alpha(conf)), qeff=qeff)
end

"""
    cell_size(lattice)

Largest perpendicular width of the cell spanned by the columns of `lattice`, i.e., the longest wavelength of a
periodic potential in the cell (the smallest reciprocal lattice vector is 2π / cell_size).
"""
cell_size(lattice) = maximum(abs(det(lattice)) / norm(cross(lattice[:, j], lattice[:, k])) for (j, k) in ((2, 3), (3, 1), (1, 2)))

"""
    apply_ewald_settings!(conf; strcs=Structure[], comm=nothing, training=false, rank=0, verbosity=get_verbosity(conf))

Ewald settings of a model, before it is built:

1. Use a fitted model with the settings it was fitted with: the settings in the header of the parameter file
   (`init_params` of the `Ewald` block, else `init_params`) that are not set in the config are taken from the file,
   e.g., for larger cells. Settings set in the config that differ from the file raise an error, unless
   `check_params = false` in the `Ewald` block. For `method = ewald`, rcut and α only affect the numerics and are not
   taken from the file.
2. Wolf and Zahn, where α = 3 / rcut (`ewald_alpha`) unless `alpha` is set: in a fit (`training`) without `rcut` and
   `alpha`, rcut = 1.5 L with the size L of the training cells (`cell_size`, largest over `strcs` and the ranks of `comm`),
   i.e. α = 2 / L, so that the potential of the longest wavelength in the cells is kept to ≥ 90 % (1 - exp(-k²/4α²) at
   k = 2π/L). With only `alpha` given, rcut = 3 / α. Both are written to the header of the parameter file at the end of
   the fit. Without a fit, rcut or alpha have to come from the file or the config.
"""
function apply_ewald_settings!(conf; strcs=Structure[], comm=nothing, training=false, rank=0, verbosity=get_verbosity(conf))
    adopt_ewald_settings!(conf, rank=rank, verbosity=verbosity)
    lowercase(get_ewald_method(conf)) ∈ ("wolf", "zahn") || return nothing
    set_rcut, set_alpha = conf("rcut", "Ewald") ≠ "default", conf("alpha", "Ewald") ≠ "default"
    (set_alpha && set_rcut) && return nothing
    if set_alpha
        set_value!(conf, "rcut", "Ewald", 3 / get_ewald_alpha(conf))
        return nothing
    end
    set_rcut && return nothing
    if !training
        error("Ewald method $(get_ewald_method(conf)) without rcut and alpha: they are set from the size of the training " *
              "cells in a fit and written to its params.dat. Use the params.dat of a fit (init_params) or set rcut or alpha " *
              "in the Ewald block.")
    end
    L = isempty(strcs) ? 0.0 : maximum(cell_size(strc.lattice) for strc in strcs)
    isnothing(comm) || (L = MPI.Allreduce(L, max, comm))
    set_value!(conf, "rcut", "Ewald", 1.5 * L)
    if rank == 0 && verbosity > 0
        @printf("Wolf range from the training cells (L = %.3f Å): rcut = 1.5 L = %.3f Å, alpha = 3 / rcut = %.4f 1/Å\n", L, 1.5 * L, 2 / L)
    end
    return nothing
end

function adopt_ewald_settings!(conf; rank=0, verbosity=get_verbosity(conf))
    get_ewald_check_params(conf) || return nothing
    file = get_ewald_init_params(conf) ≠ "default" ? get_ewald_init_params(conf) : string(get_init_params(conf))
    isfile(file) || return nothing
    header = read_params(file)[5]
    haskey(header, "ewald_method") || return nothing

    settings = [("method", "Ewald", header["ewald_method"])]
    if lowercase(header["ewald_method"]) ∈ ("wolf", "zahn")
        append!(settings, [("rcut", "Ewald", header["ewald_rcut"]), ("alpha", "Ewald", header["ewald_alpha"])])
    end
    for (key, value) in header
        startswith(key, "ewald_qeff_") && push!(settings, ("qeff", key[12:end], value))
    end

    same(a, b) = (x = tryparse(Float64, string(a)); y = tryparse(Float64, string(b));
                  isnothing(x) || isnothing(y) ? lowercase(string(a)) == lowercase(string(b)) : isapprox(x, y, rtol=1e-10))
    adopted, differences = String[], String[]
    for (key, block, value) in settings
        name = block == "Ewald" ? key : "$(key) ($block)"
        if conf(key, block) == "default"
            set_value!(conf, key, block, value)
            push!(adopted, "$name = $value")
        elseif !same(conf(key, block), value)
            push!(differences, "$name: $(conf(key, block)) in the config, $value in $file")
        end
    end
    if !isempty(differences)
        error("The Ewald settings in the config differ from the parameter file $file, with which the model was fitted:\n  " *
              join(differences, "\n  ") * "\nRemove them from the config to use the settings of the file, or set " *
              "check_params = false in the Ewald block to use the config.")
    end
    if rank == 0 && verbosity > 0 && !isempty(adopted)
        println("Ewald settings from $file: ", join(adopted, ", "))
    end
    return nothing
end

"""
    init_ewald_params(param_labels, conf; initas=get_ewald_init_params(conf))

Initialize the Ewald coupling γ of each species in `param_labels`: from the `ewald_<element>` lines of the parameter
file `initas` (e.g. the `params.dat` of a fit) if given, otherwise (and for species missing in the file) with the
`charge_scale` of the atom block, which defaults to the `charge_scale` of the `Ewald` block.
"""
function init_ewald_params(param_labels, conf; initas=get_ewald_init_params(conf))
    params = [get_charge_scale(conf, number_to_element(label)) for label in param_labels]
    if initas ≠ "default"
        _, _, _, _, _, ewald_types, ewald_parameters = read_params(initas)
        γ = Dict(zip(ewald_types, ewald_parameters))
        for (p, label) in enumerate(param_labels)
            params[p] = get(γ, number_to_element(label), params[p])
        end
    end
    return params
end

get_params(ewald::EwaldOnsites) = ewald.params

"""
    update!(ewald::EwaldOnsites, opt, dparams)

Update the Ewald onsite parameters in-place using the given optimizer
and parameter gradients `dparams`.
"""
function update!(ewald::EwaldOnsites, opt, dparams)
    update!(opt, ewald.params, dparams)
end

"""
    set_params!(ewald::EwaldOnsites, params)

Set the parameter vector of `ewald` in-place.

Throws an error if `params` does not match the expected size.
"""
function set_params!(ewald::EwaldOnsites, params)
    throw_error = size(ewald.params) ≠ size(params)
    if throw_error
        error("Parameter vector is not of correct size!")
    else
        ewald.params = params
    end
end

"""
    copy_params!(receiving_model::EwaldOnsites, sending_model::EwaldOnsites) -> Nothing

Copy matching parameters from one Ewald model to another.

# Arguments
- `receiving_model::EwaldOnsites`: The model whose parameters will be updated.
- `sending_model::EwaldOnsites`: The model providing parameter values.
"""
function copy_params!(receiving_model::EwaldOnsites, sending_model::EwaldOnsites)
    for (i, sending_label) in enumerate(sending_model.param_labels)
        for (j, receiving_label) in enumerate(receiving_model.param_labels)
            if sending_label == receiving_label
                receiving_model.params[j] = sending_model.params[i]
            end
        end
    end
end

"""
    get_hr(ewald::EwaldOnsites, sp_mode, index; apply_soc=false)

Construct the onsite Hamiltonian contribution from Ewald potentials for
the structure at `index`.

Returns a vector of sparse matrices `Hr`, where only the on-site block
(`R = 0`) contains the diagonal electrostatic shifts and all other blocks
are zero.

# Arguments
- `sp_mode`: Flag for sparse mode (not relevant)
- `index`: Structure index
- `apply_soc`: If `true`, doubles the number of orbitals (spin degree of freedom)
"""
function get_hr(ewald::EwaldOnsites, sp_mode, index; apply_soc=false)
    vals = Float64[]
    Norb = apply_soc ? 2 .* ewald.norb_per_strc[index] : ewald.norb_per_strc[index]
    for (i, ϕ_i) in enumerate(ewald.potentials[index])

        p = findfirst(label -> label == ewald.types_per_strc[index][i], ewald.param_labels)
        append!(vals, fill(-ewald.params[p]*(ϕ_i), Norb[i]))
    end

    is = collect(1:length(vals))
    Mewald = sparse(is, is, vals)
    Mzero = spzeros(ComplexF64, size(Mewald, 1), size(Mewald, 2))
    Hr = [ifelse(R == ewald.Rs_info[2, index], Mewald, Mzero) for R in 1:ewald.Rs_info[1, index]]
    return Hr
end

"""
    get_model_gradient(ewald::EwaldOnsites, indices, reg, dL_dHr; soc=true)

Gradient of the loss with respect to the Ewald coupling γ of each species. The onsite shifts are
`Δε = -γ_s φ_i` on all orbitals of atom `i` (species `s`), so `dL/dγ_s = -Σ_i Σ_o Re(dL/dHr[R=0]_oo) φ_i`.
"""
function get_model_gradient(ewald::EwaldOnsites, indices, reg, dL_dHr; soc=true)
    dparams = zeros(length(ewald.params))
    if ewald.update
        for (n, index) in enumerate(indices)
            G = dL_dHr[n][ewald.Rs_info[2, index]]
            Norb = soc ? 2 .* ewald.norb_per_strc[index] : ewald.norb_per_strc[index]
            offset = 0
            for (iion, ϕ) in enumerate(ewald.potentials[index])
                p = findfirst(==(ewald.types_per_strc[index][iion]), ewald.param_labels)
                for o in offset + 1:offset + Norb[iion]
                    dparams[p] -= real(G[o, o]) * ϕ
                end
                offset += Norb[iion]
            end
        end
    end
    return dparams + backward(reg, ewald.params)
end

"""
    write_params(ewald::EwaldOnsites, conf=get_empty_config(); filename="params.dat")

Add the Ewald coupling per species as `ewald_<element>` lines to the parameter file `filename` (written by the TB and
SOC models before), where it is read with `init_params = params.dat` in the `Ewald` block, and the Ewald settings
(`ewald_settings`) to its header.
"""
function write_params(ewald::EwaldOnsites, conf=get_empty_config(); filename="params.dat")
    isfile(filename) || return nothing
    parameters, parameter_values, ion_types, soc_parameters, _ = read_params(filename)
    write_params(parameters, parameter_values, ion_types, soc_parameters, conf, filename=replace(filename, r"\.dat$" => ""),
                 ewald_types=number_to_element.(ewald.param_labels), ewald_parameters=ewald.params, ewald_settings=ewald_settings(conf))
end