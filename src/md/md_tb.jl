"""
    MDTBModel

TB part of the Hamster Hamiltonian as a smooth function of the atomic positions, for energies and forces in the MD.

The matrix elements are the ones of the geometry tensor (`get_geometry_tensor`) times the TB parameters: for a pair of
ions at the bond vector d, `H_ab(d) = Σ_k radial_k(|d|) fcut(|d|) P_abk(d / |d|)` with the distance dependence
`radial_k` of overlap `k` (splines of `rllm.dat`) and the angular polynomials `P_abk` from `fit_angular_class`.

The angular polynomials are the exact Slater-Koster factors; the geometry tensor rounds the cosine of the bond frame
angle to five digits (`calc_angle`), so the matrix elements of a standard run differ from them by up to ~10⁻⁵ relative.

As in the MD mode of Hamster, the pairs are the ones of the reference structure (distance ≤ `rcut`), while the matrix
elements are evaluated at the actual positions. Pairs that are farther than `rcut` in the reference structure are
never included, because their parameters are usually not fitted.

# Fields
- `types`, `norb`, `offset`: type, number of orbitals (without spin) and orbital offset of every atom.
- `onsite`: constant onsite block of every type (TB onsite parameters), `soc`: onsite SOC block (with spin) per type.
- `classes`: angular polynomials of the pair types; `rllms`: distance dependence of the overlaps.
- `pi`, `pj`, `pn`, `pclass`, `pfirst`: pairs of the reference structure (sorted by `pi`) and their classes; `prev`:
  index of the reversed pair (j, i, -n).
- `bstart`: start of the block of every pair in a flattened buffer of all blocks (`nblock` values).
"""
struct MDTBModel{S}
    types :: Vector{UInt8}
    norb :: Vector{Int}
    offset :: Vector{Int}
    onsite :: Dict{UInt8, Matrix{Float64}}
    soc :: Dict{UInt8, Matrix{ComplexF64}}
    classes :: Vector{AngularClass}
    rllms :: Vector{S}
    rcut :: Float64
    rcut_tol :: Float64
    pi :: Vector{Int32}
    pj :: Vector{Int32}
    pn :: Vector{SVector{3, Int32}}
    pclass :: Vector{Int32}
    pfirst :: Vector{Int64}
    prev :: Vector{Int32}
    bstart :: Vector{Int64}
    nblock :: Int
    lattice :: SMatrix{3, 3, Float64, 9}
    reference :: Matrix{Float64}
end

nspin(tb::MDTBModel) = isempty(tb.soc) ? 1 : 2
norbitals(tb::MDTBModel) = sum(tb.norb)
npairs(tb::MDTBModel) = length(tb.pi)

"""
    md_fcut(r, rcut, rcut_tol)

Cutoff function of the TB matrix elements and its derivative: `fcut` of Hamster for `rcut_tol ≥ 0` (cosine from 1 at
`rcut` to 0 at `rcut + rcut_tol`); for `rcut_tol < 0` from 1 at `rcut - |rcut_tol|` to 0 at `rcut` and 0 beyond (Hamster
uses 1 beyond `rcut`, which only occurs if an atom has moved beyond `rcut` of a reference neighbor).
"""
function md_fcut(r, rcut, rcut_tol)
    if rcut_tol > 0
        r ≤ rcut && return 1.0, 0.0
        x = r - rcut
        x ≥ rcut_tol && return 0.0, 0.0
        return 0.5 * (cos(π * x / rcut_tol) + 1), -π / (2rcut_tol) * sin(π * x / rcut_tol)
    elseif rcut_tol < 0
        w = abs(rcut_tol)
        r ≤ rcut - w && return 1.0, 0.0
        r ≥ rcut && return 0.0, 0.0
        x = r - rcut + w
        return 0.5 * (cos(π * x / w) + 1), -π / (2w) * sin(π * x / w)
    end
    return 1.0, 0.0
end

"""
    read_tb_params(basis, conf; initas=get_init_params(conf))

TB parameters of `basis` from the parameter file `initas` (as `init_params!` of a `TBModel`).
"""
function read_tb_params(basis, conf=get_empty_config(); initas=get_init_params(conf))
    params = ones(length(basis.parameters))
    initas[1] == 'z' && return zeros(length(params))
    (initas[1] == 'o' || !isfile(initas)) && return params
    labels, values, _, _, conf_values = read_params(initas)
    check_consistency(conf_values, conf)
    for (v1, label) in enumerate(basis.parameters), (v2, file_label) in enumerate(labels)
        label == file_label && (params[v1] = values[v2])
    end
    return params
end

"""
    MDTBModel(strc, conf, comm=nothing; verbosity=get_verbosity(conf))

TB model of the MD for the reference structure `strc` (e.g. the POSCAR of the MD cell), with the orbitals, parameters
(`init_params`) and distance dependence (`rllm_file`) of the config, and SOC if `soc = true`.
"""
function MDTBModel(strc::Structure, conf=get_empty_config(), comm=nothing; verbosity=get_verbosity(conf))
    get_sepNN(conf) && error("The MD needs sepNN = false: the separate nearest-neighbor parameters switch discontinuously with the distance.")
    basis = Basis(strc, conf)
    rllm_dict = get_rllm(basis.overlaps, conf, comm=comm, verbosity=0)
    ctx = GeometryContext(strc, basis, rllm_dict, tmethod=get_tmethod(conf))
    params = read_tb_params(basis, conf)
    types = get_ion_types(strc.ions)
    norb = length.(basis.orbitals)
    offset = cumsum(vcat(0, norb[1:end-1]))
    rcut, rcut_tol = get_rcut(conf), get_rcut_tol(conf)
    lattice = SMatrix{3, 3, Float64}(strc.lattice)
    reference = reduce(hcat, [Vector(ion.pos - ion.dist) for ion in strc.ions])

    # One representative ion per type; all ions of a type must have the same orbitals
    rep = Dict{UInt8, Int}()
    for (a, t) in enumerate(types)
        r = get!(rep, t, a)
        if ctx.orbital_axes[a] ≠ ctx.orbital_axes[r] || ctx.orbital_ls[a] ≠ ctx.orbital_ls[r]
            error("The MD needs the same orbitals (and axes) on all ions of a type, which ion $a of type $t does not have.")
        end
    end

    # Constant onsite blocks, as in get_geometry_tensor for an ion with itself (r = 0)
    onsite = Dict{UInt8, Matrix{Float64}}()
    for (t, a) in rep
        block = zeros(norb[a], norb[a])
        norb[a] == 0 && (onsite[t] = block; continue)
        types_sorted = SVector{2, UInt8}(t, t)
        f = fcut(0.0, rcut, rcut_tol)
        radial = [ctx.overlap_types[k] == types_sorted ? ctx.rllms[k](0.0) * f : 0.0 for k in eachindex(basis.overlaps)]
        nnlabel = get_nn_label(0.0, 0.0, conf, onsite=get_onsite(conf), sepNN=false)
        r⃗ = SVector{3, Float64}(reference[:, a])
        geometry_entries!(ctx, a, a, 0, 0, r⃗, r⃗, nnlabel, radial) do v, i, j, k, hval
            v ≠ 0 && abs(hval) ≥ get_sp_tol(conf) && (block[i, j] += params[v] * hval)
        end
        onsite[t] = block
    end

    soc = Dict{UInt8, Matrix{ComplexF64}}()
    if get_soc(conf)
        soc_model = SOCModel([strc], [basis], isnothing(comm) ? MPI.COMM_WORLD : comm, conf)
        for t in keys(rep)
            norb[rep[t]] == 0 && continue
            soc[t] = get_param(soc_model, t) * soc_model.matrices[t]
        end
    end

    # Pairs of the reference structure and their angular classes. For ions of the same type, the conventions of the
    # geometry tensor depend on the order of the orbital indices (i < j, i > j, or both for an ion and its image).
    nl = NeighborList(reference, lattice, rcut + 1e-8; rmin=1e-8, include=(a, b) -> norb[a] > 0 && norb[b] > 0)
    nnlabel = get_nn_label(1.0, 0.0, conf, onsite=get_onsite(conf), sepNN=false)
    class_index = Dict{Tuple{UInt8, UInt8, Int}, Int32}()
    classes = AngularClass[]
    pclass = Vector{Int32}(undef, length(nl))
    for p in eachindex(nl.i)
        a, b = nl.i[p], nl.j[p]
        ord = types[a] ≠ types[b] ? -1 : Int(sign(a - b))
        pclass[p] = get!(class_index, (types[a], types[b], ord)) do
            ra, rb = rep[types[a]], ord == 0 ? rep[types[a]] : rep[types[b]]
            offsets = ord == -1 ? (0, 1000) : ord == 1 ? (1000, 0) : (0, 0)
            push!(classes, fit_angular_class(ctx, ra, rb, offsets..., params, nnlabel))
            length(classes)
        end
    end
    bsizes = [norb[nl.i[p]] * norb[nl.j[p]] for p in eachindex(nl.i)]
    bstart = cumsum(vcat(1, bsizes[1:end-1]))
    if verbosity > 0
        append_output_block("MD TB model:", ["Atoms", "Orbitals", "SOC", "Reference pairs", "Angular classes", "rcut", "rcut_tol"],
                            [length(types), sum(norb), get_soc(conf), length(nl), length(classes), rcut, rcut_tol])
    end
    return MDTBModel(types, norb, offset, onsite, soc, classes, ctx.rllms, rcut, rcut_tol, nl.i, nl.j, nl.n, pclass,
                     nl.first, reverse_pairs(nl), bstart, sum(bsizes; init=0), lattice, reference)
end

"""
    tb_blocks!(hblk, tb, positions)

Matrix elements of all reference pairs at `positions` (3 × N, unwrapped, same order and cell as the reference) into the
flattened block buffer `hblk` (block of pair `p` at `bstart[p]`, column-major norb_i × norb_j).
"""
function tb_blocks!(hblk, tb::MDTBModel, positions)
    L = tb.lattice
    nk = length(tb.rllms)
    tforeach(chunks(1:length(tb.pfirst)-1, n=4 * Threads.nthreads())) do atoms
        rad = zeros(nk)
        maxm = maximum(c -> length(c.exps), tb.classes; init=1)
        m = zeros(maxm); ∇m = zeros(SVector{3, Float64}, maxm)
        for a in atoms, p in tb.pfirst[a]:tb.pfirst[a+1]-1
            d = bond_vector(positions, L, tb.pi[p], tb.pj[p], tb.pn[p])
            r = norm(d)
            cls = tb.classes[tb.pclass[p]]
            nb = tb.norb[tb.pi[p]]
            b0 = tb.bstart[p] - 1
            @inbounds for q in 1:nb * tb.norb[tb.pj[p]]
                hblk[b0 + q] = 0.0
            end
            f, _ = md_fcut(r, tb.rcut, tb.rcut_tol)
            f == 0 && continue
            for k in cls.ks
                rad[k] = tb.rllms[k](r) * f
            end
            monomials!(m, ∇m, cls.exps, d / r)
            @inbounds for t in cls.terms
                val = 0.0
                for (q, c) in enumerate(t.coeffs)
                    val += c * m[q]
                end
                hblk[b0 + t.a + nb * (t.b - 1)] += rad[t.k] * val
            end
        end
    end
    return hblk
end

"""
    tb_pair_gradients!(g, tb, positions, W)

Derivatives `g[p] = Σ_ab W_ab ∂H_ab/∂d` of the weighted matrix elements of every pair with respect to its bond vector
d, for the weights `W` in the layout of the block buffer (the density matrix of the pair, times the spin factor). The
force on atom i is `+g[p]` and on atom j `-g[p]` for every pair (i, j).
"""
function tb_pair_gradients!(g, tb::MDTBModel, positions, W)
    L = tb.lattice
    nk = length(tb.rllms)
    tforeach(chunks(1:length(tb.pfirst)-1, n=4 * Threads.nthreads())) do atoms
        rad = zeros(nk); drad = zeros(nk)
        maxm = maximum(c -> length(c.exps), tb.classes; init=1)
        m = zeros(maxm); ∇m = zeros(SVector{3, Float64}, maxm)
        for a in atoms, p in tb.pfirst[a]:tb.pfirst[a+1]-1
            d = bond_vector(positions, L, tb.pi[p], tb.pj[p], tb.pn[p])
            r = norm(d); n̂ = d / r
            cls = tb.classes[tb.pclass[p]]
            nb = tb.norb[tb.pi[p]]
            b0 = tb.bstart[p] - 1
            f, df = md_fcut(r, tb.rcut, tb.rcut_tol)
            if f == 0 && df == 0
                g[p] = zero(SVector{3, Float64}); continue
            end
            for k in cls.ks
                s = tb.rllms[k]
                rad[k] = s(r) * f
                drad[k] = gradient(s, r, 1) * f + s(r) * df
            end
            monomials!(m, ∇m, cls.exps, n̂)
            gr = 0.0                                 # radial part, along n̂
            gt = zero(SVector{3, Float64})           # gradient of the polynomials w.r.t. n̂
            @inbounds for t in cls.terms
                w = W[b0 + t.a + nb * (t.b - 1)]
                w == 0 && continue
                val = 0.0; ∇val = zero(SVector{3, Float64})
                for (q, c) in enumerate(t.coeffs)
                    val += c * m[q]
                    ∇val += c * ∇m[q]
                end
                gr += w * drad[t.k] * val
                gt += (w * rad[t.k]) * ∇val
            end
            # d n̂ / d d = (I - n̂ n̂ᵀ) / r
            g[p] = gr * n̂ + (gt - dot(gt, n̂) * n̂) / r
        end
    end
    return g
end

"""
    tb_forces!(F, tb, positions, g)

Adds the forces of the pair gradients `g` (from `tb_pair_gradients!`) to `F`: `+g[p]` on atom i and `-g[p]` on atom
j of every pair, gathered per atom with the reversed pairs. Returns the virial `Σ d ⊗ g`.
"""
function tb_forces!(F, tb::MDTBModel, positions, g)
    Ws = tmap(chunks(1:length(tb.pfirst)-1, n=4 * Threads.nthreads())) do atoms
        W = zero(SMatrix{3, 3, Float64})
        for a in atoms
            Fa = zero(SVector{3, Float64})
            for p in tb.pfirst[a]:tb.pfirst[a+1]-1
                Fa += g[p] - g[tb.prev[p]]
                W += bond_vector(positions, tb.lattice, a, tb.pj[p], tb.pn[p]) * g[p]'
            end
            F[1, a] += Fa[1]; F[2, a] += Fa[2]; F[3, a] += Fa[3]
        end
        W
    end
    return sum(Ws; init=zero(SMatrix{3, 3, Float64}))
end

"""
    tb_reference_check(tb, positions; tol=0.5)

Error if an atom has moved by more than `tol` times the shortest reference distance from its reference site, in which
case the reference topology of the TB pairs is no longer meaningful.
"""
function tb_reference_check(tb::MDTBModel, positions)
    Linv = inv(tb.lattice)
    maxu = maximum(norm(min_image(SVector{3}(positions[:, a]) - SVector{3}(tb.reference[:, a]), tb.lattice, Linv)) for a in axes(positions, 2))
    return maxu
end
