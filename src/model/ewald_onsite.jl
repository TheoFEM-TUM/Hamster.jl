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
5. Subtracting these means from all site potentials

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
        update          = get_ewald_update(conf))

    types_per_strc = Vector{UInt8}[]
    norb_per_strc = Vector{Int64}[]
    potentials = Vector{Float64}[]
    Rs_info = zeros(Int64, 2, length(strcs))

    for (n, strc) in enumerate(strcs)
        pos = get_ion_positions(strc.ions, apply_distortion=true)

        ion_types = get_ion_types(strc.ions)
        charges = [get_qeff(conf, type) for type in number_to_element.(ion_types)]

        if method == "ewald"
            ewald = ewald_sum(pos, charges, strc.lattice, 
                                Rs              = strc.Rs, 
                                point_grid      = strc.point_grid, 
                                rcut            = rcut,
                                mesh_spacing    = mesh_spacing,
                                method          = method, 
                                alpha           = alpha
                            )
        else
            ewald = ewald_sum(pos, charges, strc.lattice, rcut=rcut, method=method, alpha=alpha)
        end

        ion_indices_with_orbs = collect(1:length(ion_types))
        ion_types_with_orbs = get_ion_types(strc.ions, conf, uniq=true, withorbitals=true)
        filter!(ind -> ion_types[ind] ∈ ion_types_with_orbs, ion_indices_with_orbs)

        push!(types_per_strc, ion_types[ion_indices_with_orbs])
        push!(norb_per_strc, size(bases[n])[ion_indices_with_orbs])
        push!(potentials, ewald.potentials[ion_indices_with_orbs])
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

    for n in eachindex(potentials)
        for i in eachindex(potentials[n])
            typ = types_per_strc[n][i]
            p = type_to_idx[typ]
            potentials[n][i] -= species_mean[p]
        end
    end

    return EwaldOnsites(params, param_labels, potentials, types_per_strc, norb_per_strc, Rs_info, update)
end

"""
    init_ewald_params(param_labels, conf; initas=get_ewald_init_params(conf))

Initialize Ewald onsite parameters for each species in `param_labels`: from the `ewald_<element>` lines of the parameter
file `initas` (e.g. the `params.dat` of a fit) if given, otherwise (and for species missing in the file) with the global
charge scaling factor `charge_scale` from `conf`.
"""
function init_ewald_params(param_labels, conf; initas=get_ewald_init_params(conf))
    params = ones(length(param_labels)) * get_ewald_charge_scale(conf)
    if initas ≠ "default"
        _, _, _, _, _, ewald_types, ewald_parameters = read_params(initas)
        file_params = Dict(zip(ewald_types, ewald_parameters))
        params = Float64[get(file_params, number_to_element(type), params[p]) for (p, type) in enumerate(param_labels)]
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

Compute the gradient of the loss with respect to the Ewald onsite parameters.

Returns a zero vector if `ewald.update` is `false`.
"""
function get_model_gradient(ewald::EwaldOnsites, indices, reg, dL_dHr; soc=true)
    dparams = zeros(length(ewald.params))
    ewald.update || return dparams
    for (n, index) in enumerate(indices)
        iR0 = ewald.Rs_info[2, index]
        Norb = soc ? 2 .* ewald.norb_per_strc[index] : ewald.norb_per_strc[index]
        offset = 0
        for (i, ϕ_i) in enumerate(ewald.potentials[index])
            p = findfirst(label -> label == ewald.types_per_strc[index][i], ewald.param_labels)
            for o in offset+1:offset+Norb[i]
                dparams[p] -= real(dL_dHr[n][iR0][o, o]) * ϕ_i
            end
            offset += Norb[i]
        end
    end
    return dparams + backward(reg, ewald.params)
end

"""
    write_params(ewald::EwaldOnsites, conf=get_empty_config(); filename="params.dat")

Add the Ewald couplings to the parameter file `filename` (written by the TB and SOC models before), where they are read
with `init_params = params.dat` in the `Ewald` block, and the Ewald settings (`ewald_settings`) to its header. The
coupling of a species is written as an `ewald_<element>` line.
"""
function write_params(ewald::EwaldOnsites, conf=get_empty_config(); filename="params.dat")
    isfile(filename) || return nothing
    parameters, parameter_values, ion_types, soc_parameters, _ = read_params(filename)
    ewald_types = number_to_element.(ewald.param_labels)
    write_params(parameters, parameter_values, ion_types, soc_parameters, conf, filename=replace(filename, r"\.dat$" => ""),
                 ewald_types=ewald_types, ewald_parameters=ewald.params, ewald_settings=ewald_settings(conf, types=ewald_types))
end

"""
    ewald_settings(conf; types=String[])

Settings that define the Ewald potentials: method, rcut, α and the charges `qeff`, written to the header of the parameter
file of a fit. The charges are listed for all ion types with a `qeff` tag and for the ion types `types` (e.g. the species
of the Ewald model, whose charge may come from `auto_qeff`).
"""
function ewald_settings(conf; types=String[])
    qeff_types = sort(unique(vcat([type for (type, block) in conf.blocks if haskey(block, "qeff")], collect(types))))
    return (method=lowercase(get_ewald_method(conf)), rcut=get_ewald_rcut(conf), alpha=get_ewald_alpha(conf),
            qeff=[(type, get_qeff(conf, type)) for type in qeff_types])
end