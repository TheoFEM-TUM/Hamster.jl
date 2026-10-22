"""
    MDFitResult

Result of an MD fit (or evaluation): losses for HyperOpt (`L_train`, `L_val`: force RMSE + loss_energy_weight × energy
RMSE per atom) and the errors and predictions of the data sets.
"""
struct MDFitResult
    L_train :: Vector{Float64}
    L_val :: Vector{Float64}
    errors :: Dict{String, Float64}
    predictions :: Dict{String, Any}
end

function save(res::MDFitResult, rank=0; filename="hamster_out.h5")
    rank == 0 || return nothing
    h5open(filename, "cw") do f
        f["L_train"] = res.L_train; f["L_val"] = res.L_val
        for (key, value) in res.errors
            f["errors/$key"] = value
        end
        for (key, value) in res.predictions
            f["predictions/$key"] = value
        end
    end
end

"""
    electronic_settings_key(conf, files; stride, skip)

String that identifies the electronic energies and forces of the data `files`: the TB parameters, distance dependence,
Ewald and SOC settings, electronic solver settings, and the data files with their modification times.
"""
function electronic_settings_key(conf, files; stride, skip)
    filehash(f) = isfile(f) ? string(hash(read(f)), base=16) : "none"
    parts = String[filehash(string(get_init_params(conf))), filehash(get_rllm_file(conf)), filehash(get_poscar(conf)),
                   string(get_rcut(conf)), string(get_rcut_tol(conf)), string(get_soc(conf)), string(get_md_kt(conf)),
                   string(get_md_kgrid(conf)), lowercase(get_md_electrons(conf)), string(get_md_probe_radius(conf)),
                   string(get_md_kpm_tol(conf)), string(get_md_kpm_order(conf)), string(stride), string(skip)]
    for (block, d) in sort(collect(conf.blocks), by=first)
        (block == "Ewald" || haskey(d, "orbitals") || haskey(d, "valence") || haskey(d, "qeff")) || continue
        push!(parts, block * string(sort(collect(d))))
    end
    for f in files
        push!(parts, f, string(isfile(f) ? mtime(f) : 0.0))
    end
    return string(hash(join(parts, "|")), base=16)
end

"""
    electronic_reference(pot, data, comm; rank=0, nranks=1, verbosity=1)

Electronic free energies and forces of all frames of `data` (distributed over the MPI ranks).
"""
function electronic_reference(pot::MDPotential, data::MDData, comm; rank=0, nranks=1, verbosity=1)
    Nf = nframes(data)
    N = length(data.types)
    E = zeros(Nf); F = zeros(3, N, Nf); μ = zeros(Nf)
    t = @elapsed for n in rank+1:nranks:Nf
        maximum(abs, data.lattice[:, :, n] - pot.tb.lattice) < 1e-4 ||
            error("Frame $(data.source[n]) has a different cell than the reference structure (poscar).")
        pos = unwrap_to_reference(data.positions[:, :, n], pot.tb.reference, pot.tb.lattice)
        res = md_compute!(pot, pos)
        E[n] = res.E_el; F[:, :, n] = res.forces; μ[n] = res.μ
        if verbosity > 1 && rank == 0 && (n - 1) % (10nranks) == 0
            @printf("   electrons of frame %d / %d\n", n, Nf); flush(stdout)
        end
    end
    if !isnothing(comm) && nranks > 1
        E = MPI.Allreduce(E, +, comm); F = MPI.Allreduce(F, +, comm); μ = MPI.Allreduce(μ, +, comm)
    end
    verbosity > 0 && rank == 0 && @printf("   Electronic energies and forces of %d frames in %.1f s\n", Nf, t)
    return E, F, μ
end

function cached_electronic_reference(pot, data, files, comm, conf; stride, rank=0, nranks=1, verbosity=1)
    cache = get_md_cache(conf)
    key = electronic_settings_key(conf, files; stride=stride, skip=get_md_skip(conf))
    if lowercase(cache) ≠ "none" && isfile(cache)
        found = h5open(cache, "r") do f
            haskey(f, key) ? (read(f[key]["E"]), read(f[key]["F"]), read(f[key]["mu"])) : nothing
        end
        if !isnothing(found)
            verbosity > 0 && rank == 0 && println("   Electronic energies and forces from the cache $cache")
            return found
        end
    end
    E, F, μ = electronic_reference(pot, data, comm; rank=rank, nranks=nranks, verbosity=verbosity)
    if lowercase(cache) ≠ "none" && rank == 0
        h5open(cache, isfile(cache) ? "r+" : "w") do f
            haskey(f, key) && delete_object(f, key)
            f["$key/E"] = E; f["$key/F"] = F; f["$key/mu"] = μ
            attributes(f[key])["files"] = join(files, " ")
        end
    end
    return E, F, μ
end

"""
    shortest_distances(data, L, pairs; rmax=6.0)

Shortest distance of every species pair over all frames.
"""
function shortest_distances(data::MDData, rmax=6.0)
    rmin = Dict{Tuple{UInt8, UInt8}, Float64}()
    for n in 1:nframes(data)
        nl = NeighborList(data.positions[:, :, n], data.lattice[:, :, n], rmax; rmin=1e-8)
        L = SMatrix{3, 3, Float64}(data.lattice[:, :, n])
        for p in eachindex(nl.i)
            a, b = data.types[nl.i[p]], data.types[nl.j[p]]
            P = a ≤ b ? (a, b) : (b, a)
            r = norm(bond_vector(data.positions[:, :, n], L, nl.i[p], nl.j[p], nl.n[p]))
            rmin[P] = min(get(rmin, P, Inf), r)
        end
    end
    return rmin
end

"""
    fit_repulsive!(rep, data, E_el, F_el, wolf, conf)

Linear least-squares fit of the coefficients of the nuclear term `rep` to the differences between the reference and the
electronic energies and forces of `data`: mean squared errors of the energies per atom (weight `energy_weight`) and of
the force components (weight `force_weight`), ridge regularization `lambda` and smoothness of the splines `smoothness`.
"""
function fit_repulsive!(rep::MDRepulsive, data::MDData, E_el, F_el, wolf, conf=get_empty_config())
    N = length(data.types); Nf = nframes(data)
    nc = ncoeffs(rep)
    AtA = zeros(nc, nc); Atb = zeros(nc)
    wE = get_md_energy_weight(conf) / Nf
    wF = get_md_force_weight(conf) / (3N * Nf)
    rcmax = max(rep.rcut, isnothing(wolf) ? 0.0 : wolf.rc)
    parts = tmap(1:Nf) do n
        pos = data.positions[:, :, n]; L = SMatrix{3, 3, Float64}(data.lattice[:, :, n])
        nl = NeighborList(pos, L, rcmax; rmin=1e-8)
        xE, XF, Ewall, Fwall = repulsive_features(rep, nl, pos, L, data.types; wolf=wolf)
        yE = data.energies[n] - E_el[n] - Ewall
        yF = vec(data.forces[:, :, n] .- F_el[:, :, n] .- Fwall)
        (wE / N^2 * (xE * xE') + wF * (XF' * XF), wE / N^2 * yE * xE + wF * (XF' * yF))
    end
    for (A, b) in parts
        AtA .+= A; Atb .+= b
    end
    # Regularization: ridge (tiny for the atomic energies, which only matter in combination for a fixed composition)
    # and second differences of the spline coefficients
    λ = get_md_lambda(conf); κ = get_md_smoothness(conf)
    for c in 1:nc
        AtA[c, c] += c ≤ length(rep.species) ? 1e-10 : λ
    end
    for P in eachindex(rep.pairs)
        off = spline_offset(rep, P)
        for k in 2:rep.nknots-1
            idx = off .+ (k-1:k+1); w = [1.0, -2.0, 1.0]
            AtA[idx, idx] .+= κ .* (w * w')
        end
    end
    rep.coeffs = Symmetric(AtA) \ Atb
    return rep
end

"""
    md_errors(rep, data, E_el, F_el, wolf)

Total energies and forces of the model (electrons + nuclear term) for `data`, and their errors: RMSE and MAE of the
absolute energy per atom (eV) and of the force components (eV/Å).
"""
function md_errors(rep::MDRepulsive, data::MDData, E_el, F_el, wolf)
    N = length(data.types); Nf = nframes(data)
    rcmax = max(rep.rcut, isnothing(wolf) ? 0.0 : wolf.rc)
    E = zeros(Nf); F = zeros(3, N, Nf)
    tforeach(1:Nf) do n
        pos = data.positions[:, :, n]; L = SMatrix{3, 3, Float64}(data.lattice[:, :, n])
        nl = NeighborList(pos, L, rcmax; rmin=1e-8)
        Fn = zeros(3, N)
        Erep, _ = repulsive_energy_forces!(Fn, rep, nl, pos, L, data.types; wolf=wolf)
        E[n] = E_el[n] + Erep
        F[:, :, n] = F_el[:, :, n] .+ Fn
    end
    dE = (E .- data.energies) ./ N
    dF = F .- data.forces
    return (E=E, F=F, rmse_E=sqrt(mean(abs2, dE)), mae_E=mean(abs, dE), rmse_F=sqrt(mean(abs2, dF)), mae_F=mean(abs, dF),
            rmse_F_ref=sqrt(mean(abs2, data.forces)), std_E=std(data.energies ./ N))
end

md_loss(err, conf) = err.rmse_F + get_md_loss_energy_weight(conf) * err.rmse_E

"""
    run_md_fit(comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), write_output=true)

Learn the nuclear term from the reference data (`train_data`) with the electrons of the config, validate it on
`val_data`, and write it to `model`.
"""
function run_md_fit(comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), write_output=true)
    train = load_md_data(get_md_train_data(conf), conf, stride=get_md_stride(conf))
    isnothing(train) && error("task = fit needs train_data in the MD block.")
    val = load_md_data(get_md_val_data(conf), conf, stride=get_md_val_stride(conf))
    strc = md_reference_structure(conf, supercell=[1, 1, 1])
    get_ion_types(strc.ions) == train.types || error("The atoms of the training data differ from the reference structure (poscar).")
    pot = MDPotential(strc, conf, comm, verbosity=write_output ? verbosity : 0)
    if verbosity > 0 && rank == 0
        @printf("MD fit: %d training frames, %d validation frames, %d atoms\n", nframes(train), isnothing(val) ? 0 : nframes(val), length(train.types))
        flush(stdout)
    end
    E_tr, F_tr, μ_tr = cached_electronic_reference(pot, train, get_md_train_data(conf), comm, conf, stride=get_md_stride(conf),
                                                   rank=rank, nranks=nranks, verbosity=verbosity)
    rmin_data = shortest_distances(train, get_md_rep_rcut(conf))
    margin = get_md_rep_margin(conf)
    rmin = Dict(P => (get_md_rep_rmin(conf) > 0 ? get_md_rep_rmin(conf) : r - margin) for (P, r) in rmin_data)
    coulomb = get_md_rep_coulomb(conf) && !isnothing(pot.wolf)
    rep = MDRepulsive(unique(train.types), rmin, conf, coulomb=coulomb)
    fit_repulsive!(rep, train, E_tr, F_tr, pot.wolf, conf)
    err_tr = md_errors(rep, train, E_tr, F_tr, pot.wolf)
    errors = Dict("train_rmse_E" => err_tr.rmse_E, "train_rmse_F" => err_tr.rmse_F, "train_mae_E" => err_tr.mae_E,
                  "train_mae_F" => err_tr.mae_F)
    predictions = Dict{String, Any}("train_E_dft" => train.energies, "train_E" => err_tr.E, "train_E_el" => E_tr,
                                    "train_F_dft" => train.forces, "train_F" => err_tr.F, "train_F_el" => F_tr, "train_mu" => μ_tr)
    L_val = [NaN]
    if !isnothing(val)
        E_va, F_va, μ_va = cached_electronic_reference(pot, val, get_md_val_data(conf), comm, conf, stride=get_md_val_stride(conf),
                                                       rank=rank, nranks=nranks, verbosity=verbosity)
        err_va = md_errors(rep, val, E_va, F_va, pot.wolf)
        merge!(errors, Dict("val_rmse_E" => err_va.rmse_E, "val_rmse_F" => err_va.rmse_F, "val_mae_E" => err_va.mae_E,
                            "val_mae_F" => err_va.mae_F))
        merge!(predictions, Dict("val_E_dft" => val.energies, "val_E" => err_va.E, "val_E_el" => E_va, "val_F_dft" => val.forces,
                                 "val_F" => err_va.F, "val_F_el" => F_va, "val_mu" => μ_va))
        L_val = [md_loss(err_va, conf)]
    end
    if verbosity > 0 && rank == 0
        @printf("   Training:   energy RMSE %.2f meV/atom, force RMSE %.4f eV/Å (forces: RMS %.4f eV/Å)\n",
                1e3err_tr.rmse_E, err_tr.rmse_F, err_tr.rmse_F_ref)
        haskey(errors, "val_rmse_E") && @printf("   Validation: energy RMSE %.2f meV/atom, force RMSE %.4f eV/Å\n",
                                                1e3errors["val_rmse_E"], errors["val_rmse_F"])
        flush(stdout)
    end
    if write_output && rank == 0
        meta = Dict{String, Any}("kt" => get_md_kt(conf), "electrons" => get_md_electrons(conf), "kgrid" => get_md_kgrid(conf),
                                 "train_rmse_E" => err_tr.rmse_E, "train_rmse_F" => err_tr.rmse_F)
        save_repulsive(rep, get_md_model(conf), meta=meta)
        write_repulsive_table(rep, replace(get_md_model(conf), r"\.h5$" => "") * "_pairs.dat")
        append_output_block("MD fit:", collect(keys(errors)), collect(values(errors)))
    end
    return MDFitResult([md_loss(err_tr, conf)], L_val, errors, predictions)
end

"""
    run_md_eval(comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf))

Energies and forces of the frames of `train_data` with the electrons of the config and the nuclear term `model`, and
their errors with respect to the reference data.
"""
function run_md_eval(comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf))
    data = load_md_data(get_md_train_data(conf), conf, stride=get_md_stride(conf))
    isnothing(data) && error("task = eval needs train_data (the data to evaluate) in the MD block.")
    strc = md_reference_structure(conf, supercell=[1, 1, 1])
    pot = MDPotential(strc, conf, comm, verbosity=verbosity)
    rep = load_repulsive(get_md_model(conf))
    E_el, F_el, μ = cached_electronic_reference(pot, data, get_md_train_data(conf), comm, conf, stride=get_md_stride(conf),
                                                rank=rank, nranks=nranks, verbosity=verbosity)
    err = md_errors(rep, data, E_el, F_el, pot.wolf)
    if verbosity > 0 && rank == 0
        @printf("MD eval of %d frames: energy RMSE %.2f meV/atom, force RMSE %.4f eV/Å\n", nframes(data), 1e3err.rmse_E, err.rmse_F)
    end
    errors = Dict("rmse_E" => err.rmse_E, "rmse_F" => err.rmse_F, "mae_E" => err.mae_E, "mae_F" => err.mae_F)
    rank == 0 && append_output_block("MD eval:", collect(keys(errors)), collect(values(errors)))
    preds = Dict{String, Any}("E_dft" => data.energies, "E" => err.E, "E_el" => E_el, "F_dft" => data.forces, "F" => err.F, "mu" => μ)
    return MDFitResult([md_loss(err, conf)], [NaN], errors, preds)
end
