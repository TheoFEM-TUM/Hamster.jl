module HamsterCUDAExt

using Hamster, CUDA, cuSPARSE, cuSOLVER, LinearAlgebra
using Hamster: DeviceHk, Dense, Sparse, densify, nkpoints, shift_invert_operator
using Hamster.KrylovKit: eigsolve, EigSorter

function Hamster.gpu_device(local_rank::Integer)
    CUDA.functional() || error("device = gpu was requested, but CUDA is not functional on this node.")
    CUDA.device!(mod(local_rank, length(CUDA.devices())))
    return ArrayDevice(CuArray)
end

const CuDeviceHk = DeviceHk{<:CuVector}

# heevjBatched is limited to matrices up to 32 × 32
const MAX_BATCHED_EIGEN_SIZE = 32

function Hamster.device_eigen(Hk::CuDeviceHk, ::Dense; kwargs...)
    H = densify(Hk)
    if Hk.Nε ≤ MAX_BATCHED_EIGEN_SIZE
        Es, vs = cuSOLVER.heevjBatched!('V', 'U', H)
    else
        Es = CuMatrix{Float64}(undef, Hk.Nε, nkpoints(Hk))
        for k in axes(H, 3)
            Es[:, k], _ = cuSOLVER.Xsyevd!('V', 'U', view(H, :, :, k))
        end
        vs = H
    end
    return Array(Es), vs
end

function Hamster.device_eigen(Hk::CuDeviceHk, ::Sparse; Neig=6, target=0.0, method="shift-invert")
    Es = zeros(Neig, nkpoints(Hk))
    vs = CuArray{ComplexF64}(undef, Hk.Nε, Neig, nkpoints(Hk))
    for k in axes(Hk.nzval, 2)
        nzval = Hk.nzval[:, k]
        if method == "shift-invert"
            nzval[Hk.diag_pos] .-= target
            A = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))
            if isempty(methods(shift_invert_operator))
                error("diag_method = shift-invert on the GPU requires the CUDSS extension. Load it with `using CUDSS`.")
            end
            μs, xs = eigsolve(shift_invert_operator(A), CuArray(rand(ComplexF64, Hk.Nε)), Neig, :LM, ishermitian=true, maxiter=500)
            λs = target .+ 1 ./ real.(μs)
        elseif method == "krylov-schur"
            A = CuSparseMatrixCSR{ComplexF64, Int32}(Hk.rowptr, Hk.colval, nzval, (Hk.Nε, Hk.Nε))
            λs, xs = eigsolve(x -> A * x, CuArray(rand(ComplexF64, Hk.Nε)), Neig, EigSorter(λ -> abs(target - λ), rev=false), ishermitian=true, maxiter=500)
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
