"""
    get_loss_gradient_hr(dL_dE, vs, Rs, ks, sp_mode, sp_iterator=nothing; sp_tol=1e-10)

Compute the gradient of the loss w.r.t. the real-space Hamiltonian matrix elements directly,

    dL/dHr[R][i, j] = Σ_k exp(2πik⋅R) Σ_m dL/dE[m, k] conj(Ψ[i, m, k]) Ψ[j, m, k].

# Arguments
- `dL_dE`: Gradient of the loss w.r.t. the eigenvalues, shape `(m, k)`.
- `vs`: Eigenvectors, shape `(Nε, m, k)`.
- `Rs`, `ks`: Lattice translation vectors and k-points.
- `sp_mode`: `Dense` returns dense matrices; `Sparse` evaluates the gradient only at the entries of `sp_iterator` and returns sparse matrices.
- `sp_tol`: Entries of the sparse gradient below `sp_tol` are dropped.

# Returns
- `dL_dHr`: Vector of matrices (one per lattice vector).
"""
function get_loss_gradient_hr(dL_dE, vs, Rs, ks, ::Dense, sp_iterator=nothing; sp_tol=1e-10)
    Nε, Neig, Nk = size(vs)
    phase = exp_2πi(Rs, ks)
    # W[:, :, k] = conj(V_k) * Diagonal(dL_dE[:, k]) * transpose(V_k)
    W = Array{ComplexF64, 3}(undef, Nε, Nε, Nk)
    kpoint_tmap(1:Nk, Nε) do k
        @views Vk = vs[:, :, k]
        @views mul!(W[:, :, k], conj(Vk), dL_dE[:, k] .* transpose(Vk))
    end
    G = reshape(W, Nε^2, Nk) * transpose(phase)
    return [reshape(G[:, R], Nε, Nε) for R in axes(G, 2)]
end

function get_loss_gradient_hr(dL_dE, vs, Rs, ks, ::Sparse, sp_iterator; sp_tol=1e-10)
    Nε, Neig, Nk = size(vs)
    phase = exp_2πi(Rs, ks)
    # Band index first, such that the sum over bands runs over contiguous memory
    wv = permutedims(conj.(vs) .* reshape(dL_dE, 1, Neig, Nk), (2, 1, 3))
    v = permutedims(vs, (2, 1, 3))
    # All entries (R, i, j) in equally large chunks: the lattice vectors have very different numbers of entries (most at
    # R = 0), so distributing the lattice vectors over the threads would leave most of them idle
    offsets = cumsum(vcat(0, length.(sp_iterator)))
    vals = Vector{ComplexF64}(undef, offsets[end])
    offsets[end] > 0 && tforeach(chunks(1:offsets[end], n=min(offsets[end], 4 * Threads.nthreads()))) do range
        R = searchsortedlast(offsets, first(range) - 1)
        @inbounds for e in range
            while e > offsets[R+1]; R += 1; end
            i, j = sp_iterator[R][e - offsets[R]]
            acc = zero(ComplexF64)
            for k in 1:Nk
                s = zero(ComplexF64)
                @simd for m in 1:Neig
                    s += wv[m, i, k] * v[m, j, k]
                end
                acc += phase[R, k] * s
            end
            vals[e] = acc
        end
    end
    dL_dHr = Vector{SparseMatrixCSC{ComplexF64, Int64}}(undef, length(sp_iterator))
    tforeach(eachindex(sp_iterator)) do R
        inds = sp_iterator[R]
        dL_dHr[R] = sparse(Int64[i for (i, _) in inds], Int64[j for (_, j) in inds], vals[offsets[R]+1:offsets[R+1]], Nε, Nε)
    end
    droptol!.(dL_dHr, sp_tol)
    return dL_dHr
end
