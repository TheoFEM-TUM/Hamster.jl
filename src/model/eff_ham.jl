struct EffectiveHamiltonian{T, S1, S2, IT}
    Nstrc :: Int64
    models :: T
    sp_mode :: S1
    sp_diag :: S2
    sp_tol :: Float64
    sp_iterator :: IT
    soc :: Bool
    Rs :: Vector{Matrix{Float64}}
end

function EffectiveHamiltonian(strcs, bases, comm, conf=get_empty_conf(); tb_model=get_tb_model(conf), ml_model=get_ml_model(conf), sp_mode=get_sp_mode(conf), sp_diag=get_sp_diag(conf), sp_tol=get_sp_tol(conf), soc=get_soc(conf), ml_data_points=nothing, rank=0, nranks=1, verbosity=get_verbosity(conf))
    if isempty(strcs) && isempty(bases)
        return EffectiveHamiltonian(0, nothing, Dense(), Dense(), 1e-10, Tuple{Int64, Int64, Int64}[], false, [zeros(3, 1)])
    end
    
    Rs = [strc.Rs for strc in strcs]
    sp_iterator = get_sparse_iterator(strcs[1], bases[1], conf)

    models = ()
    if tb_model
        if rank == 0 && verbosity > 1; println("   Getting TB model..."); end
        begin_time = MPI.Wtime()
        models = (models..., TBModel(strcs, bases, conf))
        tb_time = MPI.Wtime() - begin_time
        if rank == 0 && verbosity > 1; println("    TB time: $tb_time s"); end
    end
    if ml_model && tb_model
        if rank == 0 && verbosity > 1; println("   Getting ML model..."); end
        begin_time = MPI.Wtime()
        kernel = HamiltonianKernel(strcs, bases, models[1], comm, conf, rank=rank, nranks=nranks)
        if ml_data_points ≠ nothing
            kernel.data_points = ml_data_points
        end
        models = (models..., kernel)
        ml_time = MPI.Wtime() - begin_time
        if rank == 0 && verbosity > 1; println("    ML time: $ml_time s"); end
    end

    return EffectiveHamiltonian(length(strcs), models, sp_mode, sp_diag, sp_tol, sp_iterator, soc, Rs)
end

"""
    get_hamiltonian(ham::EffectiveHamiltonian, index, ks)

Construct the Hamiltonian matrix for given k-points `ks` from the real-space Hamiltonian and lattice vectors.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object.
- `index::Int`: The index of the structure for which the Hamiltonian is to be calculated.
- `ks`: The k-points for which the Hamiltonian matrix is to be calculated.

# Returns
- `Hk`: The Hamiltonian matrix in reciprocal space corresponding to the given k-points.
"""
function get_hamiltonian(ham::EffectiveHamiltonian, index, ks)
    Hr = get_hr(ham, index)
    Hk = get_hamiltonian(Hr, ham.Rs[index], ks, ham.sp_diag)
    return Hk
end

"""
    get_hr(ham::EffectiveHamiltonian, index)

Retrieve the real-space Hamiltonian (`Hr`) by combining contributions from individual models within the `EffectiveHamiltonian`.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object.
- `index::Int`: The index of the structure for which the Hamiltonian is to be calculated.

# Returns
- `Hr`: The combined real-space Hamiltonian matrix, obtained by summing the Hamiltonians of all the models in the `ham.models` tuple.
"""
function get_hr(ham::EffectiveHamiltonian, index)
    Hr = mapreduce(+, ham.models) do model
        get_hr(model, ham.sp_mode, index, apply_soc=ham.soc)
    end
    return Hr
end

"""
    update!(ham::EffectiveHamiltonian, opt, dL_dHr)

Update the parameters of each model within the `EffectiveHamiltonian` object using a provided update rule.

# Arguments
- `ham::EffectiveHamiltonian`: The effective Hamiltonian object containing multiple models.
- `opt`: An optimizer object specifying the update rule (e.g., ADAM).
- `model_grad`: The gradient of the loss w.r.t. the model parameters.

# Returns
- This function modifies the `ham` object in place, updating the parameters of each model it contains.
"""
function update!(ham::EffectiveHamiltonian, opt, model_grad)
    for (model, grad) in zip(ham.models, model_grad)
        update!(model, opt, grad)
    end
end

"""
    copy_params!(receiving_ham::H1, sending_ham::H2) where {H1,H2<:EffectiveHamiltonian}

Copy parameters from one EffectiveHamiltonian (`sending_ham`) to another (`receiving_ham`).

# Arguments
- `receiving_ham`: The Hamiltonian object that will receive the parameters.
- `sending_ham`: The Hamiltonian object providing the parameters to be copied.
"""
function copy_params!(receiving_ham::H1, sending_ham::H2) where {H1,H2<:EffectiveHamiltonian}
    for (receiving_model, sending_model) in zip(receiving_ham.models, sending_ham.models) 
        set_params!(receiving_model, get_params(sending_model))
    end
end

"""
    write_params(eff_ham::EffectiveHamiltonian, conf)

Writes the parameters of all models contained within an `EffectiveHamiltonian`.

# Arguments
- `eff_ham::EffectiveHamiltonian`: An EffectiveHamiltonian model.
- `conf::Config`: A configuration object.

# Behavior
- Iterates through all models within `eff_ham` and calls `write_params(model)` for each.
"""
function write_params(eff_ham::EffectiveHamiltonian, conf=get_empty_config())
    for model in eff_ham.models
        write_params(model, conf)
    end
end

function get_ml_data_points(eff_ham, conf=get_empty_config(); tb_model=get_tb_model(conf), ml_model=get_ml_model(conf))
    if tb_model && ml_model
        return eff_ham.models[2].data_points
    else
        return nothing
    end
end

"""
    get_sparse_iterator(strc, basis, conf=get_empty_config(), rcut=get_rcut(conf))

Constructs an iterator that generates sparse matrix indices for interactions within a specified cutoff radius.

# Arguments
- `strc`: A structure object containing information about ions, positions, lattice, and grid points.
- `basis`: A basis object, containing the orbitals associated with each ion.
- `conf`: Configuration object, which contains relevant simulation parameters.
- `rcut`: A cutoff radius, that defines the maximum distance for interactions to be included.

# Returns
- `indices`: A vector of vectors, where each sub-vector corresponds to a specific grid point (denoted by `R`) 
  and contains tuples `(i, j)` representing the sparse matrix indices for interactions within the cutoff radius.
"""
function get_sparse_iterator(strc, basis, conf=get_empty_config(), rcut=get_rcut(conf))
    nn_grid_points = iterate_nn_grid_points(strc.point_grid)
    Norb_per_ion = length.(basis.orbitals)
    ij_map = get_ion_orb_to_index_map(Norb_per_ion)
    Ts = frac_to_cart(strc.Rs, strc.lattice)
    indices = [Tuple{Int64, Int64}[] for R in axes(strc.Rs, 2)]
    
    for (iion, jion, R) in nn_grid_points
        r⃗₁ = strc.ions[iion].pos - strc.ions[jion].dist
        r⃗₂ = strc.ions[jion].pos - strc.ions[jion].dist - Ts[:, R]
        r = normdiff(r⃗₁, r⃗₂)
        if r ≤ rcut
            for iorb in 1:Norb_per_ion[iion], jorb in 1:Norb_per_ion[jion]
                i = ij_map[(iion, iorb)]
                j = ij_map[(jion, jorb)]
                push!(indices[R], (i, j))
            end
        end
    end
    return indices
end