import Hamster: get_sk_transform_matrix


# test if obtained system is orthonormal
function is_orthonormal(Ê; ε=1e-10)
    e1 = Ê[1, :]; e2 = Ê[2, :]; e3 = Ê[3, :]
    t1 = e1 ⋅ e2 < ε; t2 = e1 ⋅ e3 < ε; t3 = e2 ⋅ e3 < ε
    t4 = 1 - ε < norm(e1) < 1 + ε
    t5 = 1 - ε < norm(e2) < 1 + ε
    t6 = 1 - ε < norm(e3) < 1 + ε
    if (t1 && t2 && t3 && t4 && t5 && t6) == false;
            throw("System not orthonormal!"); end
    return t1 && t2 && t3 && t4 && t5 && t6
end

# check that system is righthanded
function is_righthanded(Ê; ε=1e-10)
    t1 = norm(Ê[1, :] × Ê[2, :] .- Ê[3, :]) < ε
    if t1 == false; throw("System is not righthanded!"); end
    return t1
end

# check if z-axis is along connecting vector
test_zaxis(Ê, r⃗₁, r⃗₂; ε=1e-4) = norm(Ê[3, :] .- normalize(r⃗₂ .- r⃗₁)) < ε

# special case when both vectors are 0
function test_special_cases(; ε=1e-10)
    Ê = get_sk_transform_matrix(Float64[0, 0, 0], Float64[0, 0, 0])
    return norm(Ê[3, :] .- [0, 0, 1]) < ε && is_orthonormal(Ê)
end

@testset "TB transform" begin
    r⃗₁ = 2 .* rand(Float64, 3) .- 1; r⃗₂ = 2 .* rand(Float64, 3) .- 1
    Ê = get_sk_transform_matrix(r⃗₁, r⃗₂)
    @test is_orthonormal(Ê)
    @test is_righthanded(Ê)
    @test test_zaxis(Ê, r⃗₁, r⃗₂)
    @test test_special_cases()
    # static vectors, as in the geometry tensor
    @test get_sk_transform_matrix(SVector{3}(r⃗₁), SVector{3}(r⃗₂)) ≈ Ê
end
