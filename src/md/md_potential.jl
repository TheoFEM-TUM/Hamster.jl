"""
    MDPotential

Potential energy surface of the MD: the Mermin free energy of the electrons in the Hamster Hamiltonian (TB, Wolf onsite
corrections, SOC) plus the nuclear term (`MDRepulsive`), with forces and virial.

# Fields
- `tb`, `pat`: TB model and Γ pattern of the Hamiltonian; `wolf`: onsite electrostatics (or `nothing`).
- `rep`: nuclear term (or `nothing` for the electronic part alone, e.g. in a fit).
- `solver`: `ExactSolver` or `KPMSolver`; `nelec`: number of electrons; `kT`: electronic temperature (eV).
- `nl_wolf`, `nl_rep`: neighbor lists (with skin) of the distance-dependent terms.
"""
mutable struct MDPotential{S<:MDElectronSolver, P}
    tb :: MDTBModel
    pat :: GammaPattern{P}
    wolf :: Union{Nothing, MDWolf}
    rep :: Union{Nothing, MDRepulsive}
    solver :: S
    nelec :: Float64
    kT :: Float64
    masses :: Vector{Float64}
    skin :: Float64
    nl_wolf :: Union{Nothing, NeighborList}
    nl_rep :: Union{Nothing, NeighborList}
    hblk :: Vector{Float64}
    g :: Vector{SVector{3, Float64}}
    nz :: Vector{P}
end

"""
    MDResult

Energies (eV), forces (eV/Å) and virial `Σ d ⊗ ∂E/∂d` (eV) of a structure: total potential energy `E`, electronic free
energy `E_el`, nuclear term `E_rep`, chemical potential `μ`, electrons per atom `w` and eigenvalues (if requested).
"""
struct MDResult
    E :: Float64
    E_el :: Float64
    E_rep :: Float64
    μ :: Float64
    forces :: Matrix{Float64}
    virial :: SMatrix{3, 3, Float64, 9}
    w :: Vector{Float64}
    eigenvalues :: Matrix{Float64}
end

"""
    md_nelec(types, conf)

Number of electrons of the TB bands: the `valence` of the atom blocks of all atoms minus the `charge` of the MD block.
"""
function md_nelec(types, conf)
    nelec = sum(get_valence(conf, number_to_element(t)) for t in types) - get_md_charge(conf)
    nelec > 0 || error("No electrons in the MD cell: set `valence` (electrons per atom in the TB bands) in the atom blocks.")
    return nelec
end

"""
    md_reference_structure(conf; supercell=get_md_supercell(conf))

Reference structure of the MD (`poscar` of the config), repeated `supercell` times along the lattice vectors.
"""
function md_reference_structure(conf=get_empty_config(); supercell=get_md_supercell(conf), poscar=get_poscar(conf))
    pc = read_poscar(poscar)
    lattice = pc.lattice
    frac = pc.rs_atom
    types = pc.atom_types
    if any(supercell .≠ 1)
        n1, n2, n3 = supercell
        frac = reduce(hcat, [(frac .+ [i, j, k]) ./ [n1, n2, n3] for i in 0:n1-1, j in 0:n2-1, k in 0:n3-1])
        # Atoms ordered as in the cell, cell by cell
        types = repeat(types, n1 * n2 * n3)
        lattice = lattice .* [n1 n2 n3]
    end
    rs = frac_to_cart(frac, lattice)
    Rs = get_translation_vectors(rs, lattice, rcut=get_rcut(conf))
    return Structure(Rs, rs, types, lattice, conf)
end

"""
    MDPotential(strc, conf, comm=nothing; rep=nothing, electrons=get_md_electrons(conf), device=init_device(conf, comm))

Potential of the MD for the reference structure `strc`, with the nuclear term `rep`.
"""
function MDPotential(strc::Structure, conf=get_empty_config(), comm=nothing; rep=nothing, electrons=get_md_electrons(conf),
                     device=init_device(conf, comm), verbosity=get_verbosity(conf))
    haskey(conf, "ML") && error("The MD does not support the ML corrections (ML block) of the Hamiltonian yet: their " *
                                "derivatives with respect to the positions are not implemented. Use a TB (+ Ewald, SOC) model.")
    tb = MDTBModel(strc, conf, comm, verbosity=verbosity)
    pat = GammaPattern(tb)
    wolf = nothing
    if get_ewald(conf)
        apply_ewald_settings!(conf, verbosity=verbosity)
        wolf = MDWolf(tb.types, tb.norb, conf)
    end
    electrons = lowercase(electrons)
    if electrons == "auto"
        electrons = pat.N ≤ get_md_exact_max_orbitals(conf) ? "exact" : "kpm"
    end
    solver = electrons == "exact" ? ExactSolver(get_md_kgrid(conf)) :
             electrons == "kpm" ? KPMSolver(tb, conf, device=device) :
             error("Unknown electrons = $electrons in the MD block (exact, kpm or auto).")
    if verbosity > 0
        names = ["Electrons", "Solver", "kT (eV)", "Orbitals (with spin)", "Nonzeros at Γ"]
        values = Any[md_nelec(tb.types, conf), electrons, get_md_kt(conf), pat.N, length(pat.rowval)]
        if solver isa KPMSolver
            append!(names, ["Probing radius (Å)", "Probing vectors", "Device"])
            append!(values, [get_md_probe_radius(conf), solver.nvec, string(typeof(device))])
        else
            push!(names, "k-points"); push!(values, length(solver.ks))
        end
        append_output_block("MD potential:", names, values)
    end
    masses = atomic_mass.(tb.types)
    return MDPotential(tb, pat, wolf, rep, solver, md_nelec(tb.types, conf), get_md_kt(conf), masses, get_md_skin(conf),
                       nothing, nothing, zeros(tb.nblock), zeros(SVector{3, Float64}, npairs(tb)), zeros(eltype(pat), length(pat.rowval)))
end

"""
    md_compute!(pot, positions; neig=0)

Energy, forces and virial of `positions` (3 × N, Cartesian, unwrapped with respect to the reference structure, i.e. every
atom near its reference site or moved there continuously), and eigenvalues if `neig > 0` (see `solve_electrons!`).
"""
function md_compute!(pot::MDPotential, positions; neig=0)
    tb = pot.tb
    L = tb.lattice
    N = length(tb.types)
    F = zeros(3, N)
    tb_blocks!(pot.hblk, tb, positions)

    Δε = zeros(N)
    if !isnothing(pot.wolf)
        pot.nl_wolf = isnothing(pot.nl_wolf) ? NeighborList(positions, L, pot.wolf.rc; skin=pot.skin, rmin=1e-8) :
                      update_neighbors(pot.nl_wolf, positions, L; rmin=1e-8)
        φ = wolf_potentials!(zeros(N), pot.wolf, pot.nl_wolf, positions, L)
        Δε = wolf_onsites(pot.wolf, φ)
    end
    assemble_gamma!(pot.nz, pot.pat, tb, pot.hblk, Δε)
    el = solve_electrons!(pot.solver, pot.pat, tb, pot.hblk, Δε, pot.nz, pot.nelec, pot.kT; eigenvalues=neig)

    tb_pair_gradients!(pot.g, tb, positions, el.W)
    W = tb_forces!(F, tb, positions, pot.g)
    if !isnothing(pot.wolf)
        W += wolf_forces!(F, pot.wolf, pot.nl_wolf, positions, L, el.w)
    end

    E_rep = 0.0
    if !isnothing(pot.rep)
        pot.nl_rep = isnothing(pot.nl_rep) ? NeighborList(positions, L, pot.rep.rcut; skin=pot.skin, rmin=1e-8) :
                     update_neighbors(pot.nl_rep, positions, L; rmin=1e-8)
        E_rep, Wrep = repulsive_energy_forces!(F, pot.rep, pot.nl_rep, positions, L, tb.types; wolf=pot.wolf,
                                               nl_coulomb=isnothing(pot.nl_wolf) ? pot.nl_rep : pot.nl_wolf)
        W += Wrep
    end
    return MDResult(el.F + E_rep, el.F, E_rep, el.μ, F, W, el.w, el.eigenvalues)
end

"""
    unwrap_to_reference(positions, reference, lattice)

Positions of a frame (e.g. wrapped into the cell by the DFT code) as the reference positions plus the shortest
displacements, the convention of `md_compute!`.
"""
function unwrap_to_reference(positions, reference, lattice)
    L = SMatrix{3, 3, Float64}(lattice); Linv = inv(L)
    out = similar(reference)
    for a in axes(reference, 2)
        r0 = SVector{3}(reference[:, a])
        out[:, a] = r0 + min_image(SVector{3}(positions[:, a]) - r0, L, Linv)
    end
    return out
end
