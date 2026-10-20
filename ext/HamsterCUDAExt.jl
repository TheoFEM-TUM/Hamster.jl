module HamsterCUDAExt

using Hamster, CUDA, cuSPARSE, cuSOLVER, LinearAlgebra
using Hamster: DeviceHk, Dense, Sparse, densify, nkpoints, shift_invert_operator, release_operator!
using Hamster.KrylovKit: eigsolve, EigSorter

function Hamster.gpu_device(local_rank::Integer)
    CUDA.functional() || error("device = gpu was requested, but CUDA is not functional on this node.")
    CUDA.device!(mod(local_rank, length(CUDA.devices())))
    return ArrayDevice(CuArray)
end

Hamster.device_synchronize(::CuArray) = CUDA.synchronize()

const CuDeviceHk = DeviceHk{<:CuVector}

# Below this size, LAPACK on the host (threaded over k-points, including the transfers) is faster than cuSOLVER.
# Measured on an RTX 4500 Ada: the host is 17× faster at Nε = 64 and 2× at 256, equal at 512, and cuSOLVER is 4× faster
# at 1024 and 10× at 2048. Batched Jacobi (heevjBatched, Nε ≤ 32) is slower than the host for all sizes.
const MIN_GPU_EIGEN_SIZE = 512

function Hamster.device_eigen(Hk::CuDeviceHk, ::Dense; vectors=true, kwargs...)
    H = densify(Hk)
    if Hk.Nε < MIN_GPU_EIGEN_SIZE
        Hh = Array(H)
        Es = zeros(Hk.Nε, nkpoints(Hk))
        Threads.@threads for k in axes(Hh, 3)
            if vectors
                eig = eigen!(Hermitian(Hh[:, :, k]))
                Es[:, k] = eig.values
                Hh[:, :, k] = eig.vectors
            else
                Es[:, k] = eigvals!(Hermitian(Hh[:, :, k]))
            end
        end
        return Es, vectors ? copyto!(H, Hh) : similar(H, 0, Hk.Nε, nkpoints(Hk))
    else
        # Eigenvalues only ('N') skip the back-transformation of the eigenvectors
        Es = CuMatrix{Float64}(undef, Hk.Nε, nkpoints(Hk))
        for k in axes(H, 3)
            Es[:, k] = vectors ? first(cuSOLVER.Xsyevd!('V', 'U', view(H, :, :, k))) : cuSOLVER.Xsyevd!('N', 'U', view(H, :, :, k))
        end
        return Array(Es), vectors ? H : similar(H, 0, Hk.Nε, nkpoints(Hk))
    end
end

function Hamster.device_eigen(Hk::CuDeviceHk, ::Sparse; Neig=6, target=0.0, method="shift-invert", vectors=true)
    Es = zeros(Neig, nkpoints(Hk))
    vs = CuArray{ComplexF64}(undef, Hk.Nε, Neig, nkpoints(Hk))
    krylovdim = max(30, 2Neig + 10)      # the KrylovKit default (30) is too small for more than about 25 eigenvalues
    for k in axes(Hk.nzval, 2)
        nzval = Hk.nzval[:, k]
        if method ∈ ("shift-invert", "shift-invert-arpack")    # always KrylovKit with cuDSS on the GPU
            nzval[Hk.diag_pos] .-= target
            A = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))
            if isempty(methods(shift_invert_operator))
                error("diag_method = shift-invert on the GPU requires the CUDSS extension. Load it with `using CUDSS`.")
            end
            op = shift_invert_operator(A)
            μs, xs, info = eigsolve(op, CuArray(rand(ComplexF64, Hk.Nε)), Neig, :LM, ishermitian=true, maxiter=500, krylovdim=krylovdim)
            Hamster.check_convergence(info, Neig, method)
            release_operator!(op)
            λs = target .+ 1 ./ real.(μs)
        elseif method == "krylov-schur"
            A = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))
            λs, xs, info = eigsolve(x -> A * x, CuArray(rand(ComplexF64, Hk.Nε)), Neig, EigSorter(λ -> abs(target - λ), rev=false), ishermitian=true,
                                    maxiter=500, krylovdim=krylovdim)
            Hamster.check_convergence(info, Neig, method)
            λs = real.(λs)
        else
            error("Unknown diag_method '$method'. Valid options are shift-invert, shift-invert-arpack and krylov-schur.")
        end
        length(λs) ≥ Neig || error("Sparse eigensolver converged only $(length(λs)) of $Neig eigenvalues at k-point $k.")
        # Keep the Neig eigenvalues closest to the target, sorted in ascending order
        closest = sortperm(abs.(λs .- target))[1:Neig]
        perm = closest[sortperm(λs[closest])]
        Es[:, k] = λs[perm]
        for (m, n) in enumerate(perm)
            vs[:, m, k] = xs[n]
        end
    end
    return Es, vs
end


# KPM unfolding on the GPU: the rescaled Hamiltonian stays on the device (CSR), the Chebyshev recursion is one sparse
# matrix-matrix product per step (cuSPARSE), the scalar products of all vectors are the diagonal of a small GEMM
# (copied to the host, Nvec × Nvec)
function Hamster.device_kpm_unfold(Hk::CuDeviceHk, V, M)
    nzval = Hk.nzval[:, 1]
    A = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))
    x0 = CuArray(rand(Hamster.Random.MersenneTwister(1), ComplexF64, Hk.Nε))
    Emax = real(eigsolve(x -> A * x, x0, 1, :LR, ishermitian=true)[1][1])
    Emin = real(eigsolve(x -> A * x, x0, 1, :SR, ishermitian=true)[1][1])
    center, halfwidth = Hamster.kpm_interval(Emin, Emax, M)
    nzval[Hk.diag_pos] .-= center
    nzval ./= halfwidth
    H̃ = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))

    Nvec = size(V, 1)
    μ = zeros(M, Nvec)
    t_prev = CuArray(permutedims(V))                 # vectors as columns
    t_curr = H̃ * t_prev
    μ[1, :] .= real.(diag(Array(t_prev' * t_prev)))
    M ≥ 2 && (μ[2, :] .= real.(diag(Array(t_prev' * t_curr))))
    t_next = similar(t_curr)
    n = 1
    while 2n < M
        copyto!(t_next, t_prev)
        mul!(t_next, H̃, t_curr, 2, -1)               # t_next = 2 H̃ t_curr - t_prev
        μ[2n + 1, :] .= 2 .* real.(diag(Array(t_curr' * t_curr))) .- μ[1, :]
        2n + 1 < M && (μ[2n + 2, :] .= 2 .* real.(diag(Array(t_next' * t_curr))) .- μ[2, :])
        t_prev, t_curr, t_next = t_curr, t_next, t_prev
        n += 1
    end
    return μ, center, halfwidth
end

end
