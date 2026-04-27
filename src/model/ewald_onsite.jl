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
        ewald_method=get_ewald_method(conf),
        rcut = get_rcut(conf),
        update=get_ewald_update(conf))

    types_per_strc = Vector{UInt8}[]
    norb_per_strc = Vector{Int64}[]
    potentials = Vector{Float64}[]
    Rs_info = zeros(Int64, 2, length(strcs))

    for n in eachindex(strcs)
        pos = get_ion_positions(strcs[n].ions, apply_distortion=true)
        box = strcs[n].lattice

        ion_types = get_ion_types(strcs[n].ions)
        charges = [get_qeff(conf, type) for type in number_to_element.(ion_types)]
        if ewald_method == "pme"
            ewald = pme_bspline(pos, charges, box, rcut=rcut)
        end

        push!(types_per_strc, ion_types)
        push!(norb_per_strc, size(bases[n]))
        push!(potentials, ewald.potentials)
        Rs_info[1, n] = size(strcs[n].Rs, 2)
        Rs_info[2, n] = findR0(strcs[n].Rs)
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

init_ewald_params(param_labels, conf) = ones(length(param_labels)) * get_ewald_charge_scale(conf)

get_params(ewald::EwaldOnsites) = ewald.params

function update!(ewald::EwaldOnsites, opt, dparams)
    update!(opt, ewald.params, dparams)
end

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

function get_model_gradient(ewald::EwaldOnsites, indices, reg, dL_dHr; soc=true)
    dparams = zeros(length(ewald.params))
    #if ewald.update
    #    
    #end
    return dparams
end

function write_params(ewald::EwaldOnsites, conf=get_empty_config())
    #currently does nothing
end