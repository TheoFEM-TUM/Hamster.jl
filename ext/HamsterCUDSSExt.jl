module HamsterCUDSSExt

using Hamster, CUDA, cuSPARSE, CUDSS, LinearAlgebra

# LDLᴴ factorization of the (indefinite) shifted Hamiltonian H - σI with cuDSS
function Hamster.shift_invert_operator(A::CuSparseMatrixCSR)
    F = ldlt(A, view='F')
    return x -> ldiv!(similar(x), F, x)
end

end
