"""
    MDState

State of an MD run: unwrapped Cartesian `positions` (Å), `velocities` (Å/fs), the result of the last force evaluation,
the `step`, the `time` (fs) and the kinetic energy taken out by the thermostat (`thermostat_energy`, eV), which makes
`E_pot + E_kin + thermostat_energy` a conserved quantity.
"""
mutable struct MDState
    positions :: Matrix{Float64}
    velocities :: Matrix{Float64}
    result :: MDResult
    step :: Int
    time :: Float64
    thermostat_energy :: Float64
end

kinetic_energy(v, m) = sum(m[a] * (v[1, a]^2 + v[2, a]^2 + v[3, a]^2) for a in eachindex(m)) / (2EV_PER_AMU)

"""
    degrees_of_freedom(N, thermostat)

Degrees of freedom of `N` atoms: 3N - 3 with a fixed center of mass (csvr, none), 3N for Langevin.
"""
degrees_of_freedom(N, thermostat) = lowercase(thermostat) == "langevin" ? 3N : 3N - 3

temperature(K, ndof) = 2K / (ndof * KB_EV)

"""
    step_rng(seed, step)

Random number generator of a step, seeded from the seed of the run and the step, so that a restarted run continues
with the same random numbers as an uninterrupted one.
"""
step_rng(seed, step) = Xoshiro(hash((seed, step, :hamster_md)))

"""
    maxwell_boltzmann_velocities(masses, T, seed)

Velocities (Å/fs) of the Maxwell-Boltzmann distribution at `T`, without center-of-mass motion and rescaled to exactly
`T` (3N - 3 degrees of freedom).
"""
function maxwell_boltzmann_velocities(masses, T, seed)
    rng = step_rng(seed, -1)
    N = length(masses)
    v = zeros(3, N)
    T ≤ 0 && return v
    for a in 1:N
        v[:, a] = sqrt(KB_EV * T * EV_PER_AMU / masses[a]) .* randn(rng, 3)
    end
    remove_com_velocity!(v, masses)
    v .*= sqrt(T / temperature(kinetic_energy(v, masses), 3N - 3))
    return v
end

function remove_com_velocity!(v, masses)
    p = v * masses
    v .-= p ./ sum(masses)
    return v
end

"""
    csvr_rescale!(v, masses, ndof, T, dt, τ, rng)

Stochastic velocity rescaling (Bussi, Donadio, Parrinello, J. Chem. Phys. 126, 014101 (2007)) of the velocities for a
time `dt`. Returns the change of the kinetic energy.
"""
function csvr_rescale!(v, masses, ndof, T, dt, τ, rng)
    K = kinetic_energy(v, masses)
    K ≤ 0 && return 0.0
    K̄ = ndof * KB_EV * T / 2
    c = exp(-dt / τ)
    R1 = randn(rng)
    S = sum(abs2, randn(rng, ndof - 1))
    Knew = K * c + K̄ / ndof * (1 - c) * (R1^2 + S) + 2R1 * sqrt(c * (1 - c) * K * K̄ / ndof)
    α = sqrt(Knew / K)
    # sign of the rescaling (Bussi's reference implementation)
    (R1 + sqrt(c * ndof * K / ((1 - c) * K̄))) < 0 && (α = -α)
    v .*= α
    return Knew - K
end

"""
    langevin_o!(v, masses, T, dt, τ, rng)

Ornstein-Uhlenbeck part (O) of the BAOAB Langevin integrator with friction 1/τ. Returns the change of the kinetic energy.
"""
function langevin_o!(v, masses, T, dt, τ, rng)
    K = kinetic_energy(v, masses)
    c1 = exp(-dt / τ); c2 = sqrt(1 - c1^2)
    for a in eachindex(masses)
        σ = sqrt(KB_EV * T * EV_PER_AMU / masses[a])
        for x in 1:3
            v[x, a] = c1 * v[x, a] + c2 * σ * randn(rng)
        end
    end
    return kinetic_energy(v, masses) - K
end

function kick!(v, F, masses, h)
    for a in eachindex(masses), x in 1:3
        v[x, a] += h * EV_PER_AMU * F[x, a] / masses[a]
    end
    return v
end

"""
    md_step!(state, pot, settings)

One time step: velocity Verlet with the CSVR thermostat (half steps before and after), BAOAB Langevin, or plain
velocity Verlet (thermostat = none).
"""
function md_step!(state::MDState, pot::MDPotential, settings; neig=0)
    dt, T, τ, thermostat = settings.dt, settings.temperature, settings.tau, settings.thermostat
    m = pot.masses
    rng = step_rng(settings.seed, state.step + 1)
    x, v = state.positions, state.velocities
    if thermostat == "langevin"
        kick!(v, state.result.forces, m, dt / 2)
        x .+= (dt / 2) .* v
        state.thermostat_energy -= langevin_o!(v, m, T, dt, τ, rng)
        x .+= (dt / 2) .* v
        state.result = md_compute!(pot, x; neig=neig)
        kick!(v, state.result.forces, m, dt / 2)
    else
        thermostat == "csvr" && (state.thermostat_energy -= csvr_rescale!(v, m, settings.ndof, T, dt / 2, τ, rng))
        kick!(v, state.result.forces, m, dt / 2)
        x .+= dt .* v
        state.result = md_compute!(pot, x; neig=neig)
        kick!(v, state.result.forces, m, dt / 2)
        thermostat == "csvr" && (state.thermostat_energy -= csvr_rescale!(v, m, settings.ndof, T, dt / 2, τ, rng))
    end
    state.step += 1
    state.time += dt
    return state
end

"""
    pressure(result, K, volume)

Pressure (GPa) from the kinetic energy and the virial `W = Σ d ⊗ ∂E/∂d`: `P = (2K - tr W) / (3V)`.
"""
pressure(res::MDResult, K, V) = (2K - tr(res.virial)) / (3V) * EV_PER_A3_IN_GPA

# ======================================================================================================================
# Output (HDF5 with extendable datasets)
# ======================================================================================================================

"""
    h5append!(f, name, x)

Append the record `x` (scalar or array) to the dataset `name` of the open HDF5 file `f`, which grows along its last
dimension (created at the first call).
"""
function h5append!(f, name, x)
    dims = size(x)
    if !haskey(f, name)
        chunk = (dims..., max(1, min(1024, cld(2^16, max(1, prod(dims))))))
        create_dataset(f, name, datatype(eltype(x)), dataspace((dims..., 0); max_dims=(dims..., -1)); chunk=chunk)
    end
    dset = f[name]
    n = size(dset)[end]
    HDF5.set_extent_dims(dset, (dims..., n + 1))
    if isempty(dims)
        dset[n+1] = x
    else
        dset[ntuple(_ -> Colon(), length(dims))..., n+1] = x
    end
    return nothing
end

"""
    MDBuffer

Records of the thermodynamic quantities collected in memory between two writes of the output file.
"""
struct MDBuffer
    thermo :: Vector{NamedTuple}
    frames :: Vector{Tuple{Int, Float64, Matrix{Float64}, Matrix{Float64}}}
    eigs :: Vector{Tuple{Int, Matrix{Float64}}}
end
MDBuffer() = MDBuffer(NamedTuple[], Tuple{Int, Float64, Matrix{Float64}, Matrix{Float64}}[], Tuple{Int, Matrix{Float64}}[])

function thermo_record(state::MDState, pot::MDPotential, ndof, V)
    K = kinetic_energy(state.velocities, pot.masses)
    r = state.result
    return (step=state.step, time=state.time, temperature=temperature(K, ndof), E_pot=r.E, E_kin=K,
            E_cons=r.E + K + state.thermostat_energy, E_el=r.E_el, E_rep=r.E_rep, mu=r.μ, pressure=pressure(r, K, V))
end

"""
    write_md_output!(filename, buf, state, pot)

Append the buffered records to the output file and update its restart data.
"""
function write_md_output!(filename, buf::MDBuffer, state::MDState, pot::MDPotential)
    L = Matrix(pot.tb.lattice); Linv = inv(L)
    h5open(filename, isfile(filename) ? "r+" : "w") do f
        if !haskey(f, "types")
            f["types"] = Int.(pot.tb.types); f["lattice"] = L; f["masses"] = pot.masses; f["reference"] = pot.tb.reference
        end
        for rec in buf.thermo, (key, value) in pairs(rec)
            h5append!(f, "thermo/$key", value)
        end
        for (step, time, x, v) in buf.frames
            h5append!(f, "trajectory/step", step); h5append!(f, "trajectory/time", time)
            h5append!(f, "trajectory/positions", x); h5append!(f, "trajectory/velocities", v)
            # Fractional positions at the top level, as the structure files (xdatcar = md.h5) of standard Hamster runs
            h5append!(f, "positions", Linv * x)
        end
        for (step, Es) in buf.eigs
            h5append!(f, "eigenvalues/step", step); h5append!(f, "eigenvalues/values", Es)
        end
        haskey(f, "restart") && delete_object(f, "restart")
        g = create_group(f, "restart")
        g["positions"] = state.positions; g["velocities"] = state.velocities; g["step"] = state.step
        g["time"] = state.time; g["thermostat_energy"] = state.thermostat_energy
        if pot.solver isa KPMSolver
            g["mu"] = pot.solver.μ; g["bounds"] = collect(pot.solver.bounds)
        end
    end
    empty!(buf.thermo); empty!(buf.frames); empty!(buf.eigs)
    return nothing
end

"""
    read_md_restart(filename)

Positions, velocities, step, time and thermostat energy (and the kpm state) from the restart data of an MD output file.
"""
function read_md_restart(filename)
    isfile(filename) || error("restart = true, but the MD output $filename does not exist.")
    h5open(filename, "r") do f
        haskey(f, "restart") || error("The MD output $filename has no restart data.")
        g = f["restart"]
        return (positions=read(g["positions"]), velocities=read(g["velocities"]), step=read(g["step"]), time=read(g["time"]),
                thermostat_energy=read(g["thermostat_energy"]), mu=haskey(g, "mu") ? read(g["mu"]) : NaN,
                bounds=haskey(g, "bounds") ? Tuple(read(g["bounds"])) : (NaN, NaN))
    end
end

"""
    truncate_md_output!(filename, step)

Remove the records after `step` from an MD output (a run that was interrupted after its last restart point), so that
the restarted run continues the records without duplicates.
"""
function truncate_md_output!(filename, step)
    h5open(filename, "r+") do f
        for (group, key) in (("thermo", "step"), ("trajectory", "step"), ("eigenvalues", "step"))
            haskey(f, group) || continue
            steps = read(f["$group/$key"])
            n = count(≤(step), steps)
            for name in keys(f[group])
                d = f["$group/$name"]
                HDF5.set_extent_dims(d, (size(d)[1:end-1]..., n))
            end
            if group == "trajectory" && haskey(f, "positions")
                d = f["positions"]; HDF5.set_extent_dims(d, (size(d)[1:end-1]..., n))
            end
        end
    end
end

"""
    run_md(comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), device=init_device(conf, comm))

NVT molecular dynamics with the Hamster Hamiltonian and the nuclear term `model` (see the MD block), written to
`output`. Returns the final `MDState`.
"""
function run_md(comm, conf; rank=0, nranks=1, verbosity=get_verbosity(conf), device=init_device(conf, comm))
    nranks > 1 && rank == 0 && @warn "The MD runs on one MPI rank (threads and GPU); the other $(nranks - 1) ranks are idle."
    rank == 0 || return nothing
    strc = md_reference_structure(conf)
    rep = load_repulsive(get_md_model(conf))
    pot = MDPotential(strc, conf, comm, rep=rep, device=device, verbosity=verbosity)
    thermostat = lowercase(get_md_thermostat(conf))
    thermostat ∈ ("csvr", "langevin", "none") || error("Unknown thermostat $thermostat (csvr, langevin or none).")
    N = length(pot.tb.types)
    settings = (dt=get_md_dt(conf), temperature=get_md_temperature(conf), tau=get_md_tau(conf), thermostat=thermostat,
                seed=get_md_seed(conf), ndof=degrees_of_freedom(N, thermostat))
    output = get_md_output(conf)
    V = abs(det(pot.tb.lattice))

    if get_md_restart(conf)
        r = read_md_restart(output)
        truncate_md_output!(output, r.step)
        x, v = r.positions, r.velocities
        step0, time0, Eth = r.step, r.time, r.thermostat_energy
        if pot.solver isa KPMSolver
            pot.solver.μ = r.mu; pot.solver.bounds = r.bounds
        end
    else
        isfile(output) && rm(output)
        x = copy(pot.tb.reference)
        if get_md_init_poscar(conf) ≠ "default"
            pc = read_poscar(get_md_init_poscar(conf))
            x = unwrap_to_reference(frac_to_cart(pc.rs_atom, pc.lattice), pot.tb.reference, pot.tb.lattice)
        end
        v = maxwell_boltzmann_velocities(pot.masses, get_md_init_temperature(conf), settings.seed)
        thermostat == "langevin" || remove_com_velocity!(v, pot.masses)
        step0, time0, Eth = 0, 0.0, 0.0
    end
    neig = get_md_eig_every(conf) > 0 ? get_neig(conf) : 0
    t0 = @elapsed result = md_compute!(pot, x; neig=(neig > 0 && step0 % get_md_eig_every(conf) == 0) ? neig : 0)
    state = MDState(x, v, result, step0, time0, Eth)
    buf = MDBuffer()
    thermo_every, traj_every, eig_every = get_md_thermo_every(conf), get_md_traj_every(conf), get_md_eig_every(conf)
    if step0 == 0
        push!(buf.thermo, thermo_record(state, pot, settings.ndof, V))
        traj_every > 0 && push!(buf.frames, (0, 0.0, copy(x), copy(v)))
        neig > 0 && push!(buf.eigs, (0, result.eigenvalues))
        write_md_output!(output, buf, state, pot)
    end
    if verbosity > 0
        @printf("MD: %d atoms, %d orbitals, %s electrons, thermostat %s, T = %.1f K, dt = %.2f fs, steps %d → %d\n", N, pot.pat.N,
                pot.solver isa KPMSolver ? "kpm" : "exact", thermostat, settings.temperature, settings.dt, step0, get_md_nsteps(conf))
        @printf("%10s %10s %9s %16s %16s %10s %9s %8s\n", "step", "time(ps)", "T(K)", "E_pot(eV)", "E_cons(eV)", "mu(eV)", "P(GPa)", "s/step")
        flush(stdout)
    end
    write_every = traj_every > 0 ? max(traj_every, 10) : 100
    warned = false
    while state.step < get_md_nsteps(conf)
        next = state.step + 1
        ne = (eig_every > 0 && next % eig_every == 0) ? neig : 0
        tstep = @elapsed md_step!(state, pot, settings; neig=ne)
        state.step % thermo_every == 0 && push!(buf.thermo, thermo_record(state, pot, settings.ndof, V))
        traj_every > 0 && state.step % traj_every == 0 && push!(buf.frames, (state.step, state.time, copy(state.positions), copy(state.velocities)))
        ne > 0 && push!(buf.eigs, (state.step, state.result.eigenvalues))
        if !warned && state.step % 100 == 0 && tb_reference_check(pot.tb, state.positions) > 1.5
            @warn "An atom has moved by more than 1.5 Å from its reference site: the TB pairs of the reference structure may no longer describe the structure."
            warned = true
        end
        if verbosity > 0 && state.step % get_md_print_every(conf) == 0
            rec = thermo_record(state, pot, settings.ndof, V)
            @printf("%10d %10.4f %9.2f %16.6f %16.6f %10.5f %9.4f %8.3f\n", rec.step, rec.time / 1000, rec.temperature, rec.E_pot,
                    rec.E_cons, rec.mu, rec.pressure, tstep)
            flush(stdout)
        end
        (state.step % write_every == 0 || state.step == get_md_nsteps(conf)) && write_md_output!(output, buf, state, pot)
    end
    write_md_output!(output, buf, state, pot)
    if get_md_analysis(conf) && traj_every > 0
        analysis = md_analysis(output, equilibration=get_md_equilibration(conf), rmax=get_md_rdf_rmax(conf), nbins=get_md_rdf_nbins(conf))
        write_md_analysis(output, analysis)
        if verbosity > 0
            @printf("Analysis (after %d steps of equilibration): T = %.2f ± %.2f K, P = %.4f ± %.4f GPa\n", get_md_equilibration(conf),
                    analysis.T_mean, analysis.T_std, analysis.P_mean, analysis.P_std)
        end
    end
    return state
end
