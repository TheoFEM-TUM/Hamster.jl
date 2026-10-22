module HamsterCUDSSExt

using Hamster, CUDA, cuSPARSE, CUDSS, LinearAlgebra

# Shift-invert operator x -> (H - σI)⁻¹ x from the LDLᴴ factorization of the (indefinite) shifted Hamiltonian with cuDSS
struct ShiftInvert{S}
    solver::S
end
(op::ShiftInvert)(x) = ldiv!(similar(x), op.solver, x)

Hamster.shift_invert_operator(A::CuSparseMatrixCSR) = ShiftInvert(ldlt(A, view='F'))

# The factorization is stored in cuDSS memory that the Julia garbage collector does not account for, so it is released
# right after the eigensolver. Otherwise it accumulates over k-points and structures until the GPU runs out of memory.
# cuDSS allocates outside the memory pool of CUDA.jl, so the memory cached by the pool is returned to the driver as well
# (otherwise the next factorization can fail although most of the pool is unused).
function Hamster.release_operator!(op::ShiftInvert)
    CUDA.synchronize()
    finalize(op.solver.data)
    finalize(op.solver.matrix)
    GC.gc(false)
    CUDA.reclaim()
    return nothing
end

end
