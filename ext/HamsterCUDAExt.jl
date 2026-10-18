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
        if method == "shift-invert"
            nzval[Hk.diag_pos] .-= target
            A = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))
            if isempty(methods(shift_invert_operator))
                error("diag_method = shift-invert on the GPU requires the CUDSS extension. Load it with `using CUDSS`.")
            end
            op = shift_invert_operator(A)
            μs, xs = eigsolve(op, CuArray(rand(ComplexF64, Hk.Nε)), Neig, :LM, ishermitian=true, maxiter=500, krylovdim=krylovdim)
            release_operator!(op)
            λs = target .+ 1 ./ real.(μs)
        elseif method == "krylov-schur"
            A = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))
            λs, xs = eigsolve(x -> A * x, CuArray(rand(ComplexF64, Hk.Nε)), Neig, EigSorter(λ -> abs(target - λ), rev=false), ishermitian=true,
                              maxiter=500, krylovdim=krylovdim)
            λs = real.(λs)
        else
            error("Unknown diag_method '$method'. Valid options are shift-invert and krylov-schur.")
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

end
