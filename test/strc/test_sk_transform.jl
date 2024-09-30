import Hamster: get_transformed_system


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
test_zaxis(Ê, r⃗₁, r⃗₂; ε=1e-5) = norm(Ê[3, :] .- (r⃗₂ .- r⃗₁) ./ norm(r⃗₂ .- r⃗₁)) < ε

# test special case when both vectors are 0
function test_special_cases(method; ε=1e-10)
    r⃗₁ = Float64[0, 0, 0]; r⃗₂ = Float64[0, 0, 0]; ω = Float64[1, 1, 1]
    Ê = ifelse(method=="GramSchmidt", get_transformed_system(r⃗₁, r⃗₂, ω),
                                    get_transformed_system(r⃗₁, r⃗₂))
    t1 = norm(Ê[3, :] .- [0, 0, 1]) < ε
    t2 = is_orthonormal(Ê)
    return t1 && t2
end

# check special case when omega lies along connecting vector
function test_omega_along_axis()
    r⃗₁ = Float64[0, 0, 0]; r⃗₂ = Float64[1, 1, 1]; ω = Float64[1, 1, 1]
    Ê = get_transformed_system(r⃗₁, r⃗₂, ω)
    Ê₂ = get_transformed_system(r⃗₁, r⃗₂, -ω)
    t1 = is_orthonormal(Ê); t2 = is_righthanded(Ê)
    t3 = test_zaxis(Ê, r⃗₁, r⃗₂)
    t4 = is_orthonormal(Ê₂); t5 = is_righthanded(Ê₂)
    t6 = test_zaxis(Ê₂, r⃗₁, r⃗₂)
    return t1 && t2 && t3 && t4 && t5 && t6
end

@testset "TB transform" begin
    # test GramSchmidt method for obtaining rotated coordinate system
    @testset "GramSchmidt method" begin
        r⃗₁ = 2 .* rand(Float64, 3) .- 1; r⃗₂ = 2 .* rand(Float64, 3) .- 1
        ω = 2 .* rand(Float64, 3) .- 1
        Ê = get_transformed_system(r⃗₁, r⃗₂, ω)
        @test is_orthonormal(Ê)
        @test is_righthanded(Ê)
        @test test_zaxis(Ê, r⃗₁, r⃗₂)
        @test test_special_cases("GramSchmidt")
        @test test_omega_along_axis()
    end
    # test rotation method for obtaining rotated coordinate system
    @testset "Rotation method" begin
        r⃗₁ = 2 .* rand(Float64, 3) .- 1; r⃗₂ = 2 .* rand(Float64, 3) .- 1
        Ê = get_transformed_system(r⃗₁, r⃗₂)
        @test is_orthonormal(Ê)
        @test is_righthanded(Ê)
        @test test_zaxis(Ê, r⃗₁, r⃗₂)
        @test test_special_cases("Rotation")
    end
end