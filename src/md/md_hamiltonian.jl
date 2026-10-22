# Assembly of the MD Hamiltonian from the TB blocks (`tb_blocks!`), the constant onsite and SOC blocks and the
# electrostatic onsite shifts, and the conversion of density matrices to the weights of the pair gradients.
#
# Orbital order: atoms in the order of the structure, orbitals of an atom as in the basis, and with SOC the two spins of
# an orbital next to each other (`apply_spin_basis`), i.e. index ns (α - 1) + σ within an atom.

"""
    GammaPattern{T}

Sparsity pattern (CSC) of the Hamiltonian at Γ: the column of an orbital of atom j contains all orbitals of the atoms
coupled to j (its reference pairs and itself), as dense blocks.

# Fields
- `colptr`, `rowval`: CSC pattern of the `N × N` matrix (`N` orbitals including spin).
- `diagpos`: position of the diagonal element of every orbital.
- `erow`: row offset of the block of atom i within the columns of atom j, for every TB pair (i, j).
- `soff`: orbital offset (0-based, including spin) of every atom; `ns`: 1, or 2 with SOC.
- `const_nz`: constant part (onsite and SOC blocks) of the nonzeros.
"""
struct GammaPattern{T}
    colptr :: Vector{Int}
    rowval :: Vector{Int}
    N :: Int
    diagpos :: Vector{Int}
    erow :: Vector{Int}
    soff :: Vector{Int}
    ns :: Int
    const_nz :: Vector{T}
end

Base.eltype(::GammaPattern{T}) where {T} = T

function GammaPattern(tb::MDTBModel)
    ns = nspin(tb)
    T = ns == 2 ? ComplexF64 : Float64
    Nat = length(tb.types)
    nso = ns .* tb.norb
    soff = cumsum(vcat(0, nso[1:end-1]))
    N = sum(nso)
    # Atoms coupled to every atom (sorted), including itself
    coupled = [Int[a] for a in 1:Nat]
    for p in eachindex(tb.pi)
        push!(coupled[tb.pj[p]], tb.pi[p])
    end
    coupled = [sort(unique(c)) for c in coupled]
    rowoff = [Dict{Int, Int}() for _ in 1:Nat]
    collen = zeros(Int, Nat)
    for b in 1:Nat
        o = 0
        for a in coupled[b]
            rowoff[b][a] = o; o += nso[a]
        end
        collen[b] = o
    end
    colptr = ones(Int, N + 1)
    for b in 1:Nat, c in 1:nso[b]
        colptr[soff[b]+c+1] = collen[b]
    end
    cumsum!(colptr, colptr)
    rowval = Vector{Int}(undef, colptr[end] - 1)
    for b in 1:Nat, c in 1:nso[b]
        q = colptr[soff[b]+c]
        for a in coupled[b], r in 1:nso[a]
            rowval[q] = soff[a] + r; q += 1
        end
    end
    diagpos = zeros(Int, N)
    for b in 1:Nat, c in 1:nso[b]
        diagpos[soff[b]+c] = colptr[soff[b]+c] + rowoff[b][b] + c - 1
    end
    erow = [rowoff[tb.pj[p]][tb.pi[p]] for p in eachindex(tb.pi)]
    const_nz = zeros(T, length(rowval))
    for b in 1:Nat
        nso[b] == 0 && continue
        t = tb.types[b]
        block = ns == 2 ? kron(tb.onsite[t], Matrix{Float64}(I, 2, 2)) .+ get(tb.soc, t, zeros(2tb.norb[b], 2tb.norb[b])) : tb.onsite[t]
        for c in 1:nso[b], r in 1:nso[b]
            const_nz[colptr[soff[b]+c] + rowoff[b][b] + r - 1] += block[r, c]
        end
    end
    return GammaPattern{T}(colptr, rowval, N, diagpos, erow, soff, ns, const_nz)
end

"""
    assemble_gamma!(nz, pat, tb, hblk, Δε)

Nonzeros of the Hamiltonian at Γ for the TB blocks `hblk` and the onsite shifts `Δε` (per atom).
"""
function assemble_gamma!(nz, pat::GammaPattern, tb::MDTBModel, hblk, Δε)
    copyto!(nz, pat.const_nz)
    ns = pat.ns
    for a in eachindex(tb.types), c in 1:ns*tb.norb[a]
        nz[pat.diagpos[pat.soff[a]+c]] += Δε[a]
    end
    tforeach(chunks(1:length(tb.pfirst)-1, n=4 * Threads.nthreads())) do atoms
        for a in atoms, p in tb.pfirst[a]:tb.pfirst[a+1]-1
            i, j = tb.pi[p], tb.pj[p]
            ni, nj = tb.norb[i], tb.norb[j]
            b0 = tb.bstart[p] - 1
            @inbounds for β in 1:nj, σ in 1:ns
                q0 = pat.colptr[pat.soff[j] + ns * (β - 1) + σ] + pat.erow[p] + σ - 1
                for α in 1:ni
                    nz[q0 + ns * (α - 1)] += hblk[b0 + α + ni * (β - 1)]
                end
            end
        end
    end
    return nz
end

"""
    gamma_weights!(W, w_atoms, pat, tb, ρnz, s)

Weights of the pair gradients (`W`, block layout) and electrons per atom (`w_atoms`) from the density matrix `ρnz` on
the pattern (Hermitian), times the spin factor `s`.
"""
function gamma_weights!(W, w_atoms, pat::GammaPattern, tb::MDTBModel, ρnz, s)
    ns = pat.ns
    tforeach(chunks(1:length(tb.pfirst)-1, n=4 * Threads.nthreads())) do atoms
        for a in atoms, p in tb.pfirst[a]:tb.pfirst[a+1]-1
            i, j = tb.pi[p], tb.pj[p]
            ni, nj = tb.norb[i], tb.norb[j]
            b0 = tb.bstart[p] - 1
            @inbounds for β in 1:nj, α in 1:ni
                acc = 0.0
                for σ in 1:ns
                    acc += real(ρnz[pat.colptr[pat.soff[j] + ns * (β - 1) + σ] + pat.erow[p] + ns * (α - 1) + σ - 1])
                end
                W[b0 + α + ni * (β - 1)] = s * acc
            end
        end
    end
    for a in eachindex(tb.types)
        w_atoms[a] = s * sum(c -> real(ρnz[pat.diagpos[pat.soff[a]+c]]), 1:ns*tb.norb[a]; init=0.0)
    end
    return W, w_atoms
end

"""
    gamma_matrix(pat, nz)

The Hamiltonian at Γ as a `SparseMatrixCSC`.
"""
gamma_matrix(pat::GammaPattern, nz) = SparseMatrixCSC(pat.N, pat.N, pat.colptr, pat.rowval, Vector(nz))

"""
    dense_hamiltonian!(H, pat, tb, hblk, Δε, k)

Dense Hamiltonian at the k-point `k` (fractional), with the phase exp(2πi k⋅n) of the lattice translation n of every
pair (any gauge gives the same eigenvalues and density matrix in the orbital basis up to this phase).
"""
function dense_hamiltonian!(H, pat::GammaPattern, tb::MDTBModel, hblk, Δε, k)
    fill!(H, 0)
    ns = pat.ns
    for b in eachindex(tb.types), c in 1:ns*tb.norb[b]
        col = pat.soff[b] + c
        for q in pat.colptr[col]:pat.colptr[col+1]-1
            r = pat.rowval[q]
            # constant onsite blocks only (the pattern also contains the coupled atoms, whose constants are zero)
            pat.const_nz[q] ≠ 0 && (H[r, col] += pat.const_nz[q])
        end
        H[col, col] += Δε[b]
    end
    for p in eachindex(tb.pi)
        i, j = tb.pi[p], tb.pj[p]
        ni, nj = tb.norb[i], tb.norb[j]
        b0 = tb.bstart[p] - 1
        phase = eltype(H) <: Real ? cospi(2 * dot(k, tb.pn[p])) : cispi(2 * dot(k, tb.pn[p]))
        for β in 1:nj, α in 1:ni, σ in 1:ns
            H[pat.soff[i] + ns * (α - 1) + σ, pat.soff[j] + ns * (β - 1) + σ] += hblk[b0 + α + ni * (β - 1)] * phase
        end
    end
    return H
end
