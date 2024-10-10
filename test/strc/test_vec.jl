# test if frac_to_cart is inverse of cart_to_frac
function test_inverse(ε)
    Ê = rand(Float64, (3, 3))
    r⃗ = rand(Float64, 3)
    t1 = norm(r⃗ .- Hamster.frac_to_cart(Hamster.cart_to_frac(r⃗, Ê), Ê)) < ε
    t2 = norm(r⃗ .- Hamster.cart_to_frac(Hamster.frac_to_cart(r⃗, Ê), Ê)) < ε
    return t1 && t2
end

# test if difference of vectors commutes with basis transform
function test_commutation(ε)
   r⃗₁ = rand(Float64, 3)
   r⃗₂ = rand(Float64, 3)
   Ê = rand(Float64, (3, 3))

   v1 = Hamster.transform_basis(r⃗₁, Ê) .- Hamster.transform_basis(r⃗₂, Ê)
   v2 = Hamster.transform_basis(r⃗₁ .- r⃗₂, Ê)
   return norm(v1 .- v2) < ε
end

# test if transform of a set of vectors works as intended
function test_matrix_transform(ε)
    M = rand(Float64, 3, 10)
    Ê = rand(Float64, (3, 3))
    T1 = Hamster.transform_basis(M, Ê)
    T2 = zeros(Float64, size(M))
    for i in axes(M, 2)
        T2[:, i] = Hamster.transform_basis(M[:, i], Ê)
    end
    return norm(T1 .- T2) < ε
end

function test_matrixinverse(ε)
    R = rand(3, 10)
    Ê = rand(3, 3)
    t1 = norm(R .- Hamster.cart_to_frac(Hamster.frac_to_cart(R, Ê), Ê)) < ε
    T2 = similar(R)
    for i in axes(R, 2)
        T2[:, i] = R[:, i] .- Hamster.cart_to_frac(Hamster.frac_to_cart(R[:, i], Ê), Ê)
    end
    t2 = norm(T2) < ε
    return t1 && t2
end

# test if transform works for silicon
function test_silicon()
    Ê = 5.45 .* [0.5 0.5 0; 0.5 0 0.5; 0 0.5 0.5]
    r⃗ = [0.25, 0.25, 0.25]
    t1 = Hamster.frac_to_cart(r⃗, Ê) == [1.3625, 1.3625, 1.3625]
    t2 = norm(Hamster.frac_to_cart([2, 0, 0], Ê)) ≈ √2 * 5.45
    t3 = norm(Hamster.frac_to_cart([0, 2, 0], Ê)) ≈ √2 * 5.45
    t4 = norm(Hamster.frac_to_cart([0, 0, 2], Ê)) ≈ √2 * 5.45
    return t1 && t2 && t3 && t4
end

# test if spherical transform is correct
function test_transform_to_spherical(ε)
    r⃗ = rand(Float64, 3)
    r, θ, φ = Hamster.transform_to_spherical(r⃗)
    x = r*sin(θ)*cos(φ)
    y = r*sin(θ)*sin(φ)
    z = r*cos(θ)
    return norm([x, y, z] .- r⃗) < ε
end

# test if spherical transform works correctly with set of vectors
function test_matrix_spherical(ε)
    M = rand(Float64, 3, 10)
    T1 = Hamster.transform_to_spherical(M)
    T2 = zeros(Float64, size(M))
    for i in axes(T2, 2)
        T2[:, i] .= Hamster.transform_to_spherical(M[:, i])
    end
    return norm(T1 .- T2) < ε
end

@testset "lattice transforms" begin
    @test test_inverse(1e-10)
    @test test_commutation(1e-10)
    @test test_matrix_transform(1e-10)
    @test test_silicon()

    @test test_transform_to_spherical(1e-10)
    @test test_matrix_spherical(1e-10)

    v1 = 2 .* rand(3) .- 1 
    v2 = 2 .* rand(3) .- 1
    @test Hamster.normdiff(v1, v2) ≈ norm(v1 - v2)

    t = 2 .* rand(3) .- 1
    @test Hamster.normdiff(v1,v2, t) ≈ norm(v1 - v2 + t)

    dv1 = 2 .* rand(3) .- 1
    dv2 = 2 .* rand(3) .- 1

    @test Hamster.normdiff(v1, v2, dv1, dv2, t) ≈ norm(v1 - dv1 - v2 + dv2 + t)
end

@testset "Projection" begin
    import Hamster: proj
    # Test 1: Basic test for orthogonal projection
    u = [1, 0]
    v = [0, 1]
    @test proj(u, v) ≈ [0, 0] # should be 0

    # Test 2: Projection of a vector onto itself
    u = [2, 3]
    v = [2, 3]
    @test proj(u, v) ≈ v

    # Test 3: Projection of a non-orthogonal vector
    u = [1, 0]
    v = [3, 4]
    @test proj(u, v) ≈ [3, 0]

    # Test 4: Projection of a vector onto a different dimension vector
    u = [1, 2]
    v = [3, 4]
    @test proj(u, v) ≈ (11/5) * u

    # Test 5: Zero vector projection
    u = [0, 0]
    v = [1, 1]
    @test proj(u, v) == [0, 0]  # Projection onto a zero vector should return a zero vector

    # Test 6: Zero projection when v is zero
    u = [1, 2]
    v = [0, 0]
    @test proj(u, v) == [0, 0]  # Projection of a zero vector onto any vector should return a zero vector

    # Test 7: Higher dimension test (3D vectors)
    u = [1, 0, 0]
    v = [2, 3, 4]
    @test proj(u, v) ≈ [2, 0, 0]
end

@testset "Angle tests" begin
    import Hamster: calc_angle
    # Test 1: Angle between two perpendicular vectors (90 degrees or π/2 radians)
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    @test isapprox(calc_angle(v1, v2), π/2, atol=1e-5)

    # Test 2: Angle between two parallel vectors (0 degrees or 0 radians)
    v1 = [1.0, 0.0]
    v2 = [2.0, 0.0]
    @test isapprox(calc_angle(v1, v2), 0.0, atol=1e-5)

    # Test 3: Angle between two opposite vectors (180 degrees or π radians)
    v1 = [1.0, 0.0]
    v2 = [-1.0, 0.0]
    @test isapprox(calc_angle(v1, v2), π, atol=1e-5)

    # Test 4: Test with zero vector (should return 0 due to ϵ threshold)
    v1 = [0.0, 0.0]
    v2 = [1.0, 1.0]
    @test calc_angle(v1, v2) == 0.0

    # Test 5: Test with nearly identical vectors (should return a very small angle)
    v1 = [1.0, 1.0]
    v2 = [1.0001, 1.0001]
    @test isapprox(calc_angle(v1, v2), 0.0, atol=1e-5)

    # Test 6: Test with a large ϵ value (should return 0 as vectors are considered too small)
    v1 = [1e-6, 1e-6]
    v2 = [1e-6, 1e-6]
    @test calc_angle(v1, v2; ϵ=1e-4) == 0.0

    # Test 7: Test with two random vectors
    v1 = rand(3)
    v2 = rand(3)
    @test isapprox(calc_angle(v1, v2), acos(round(v1 ⋅ v2 / (norm(v1)*norm(v2)), digits=5)), atol=1e-5)

    # Test 8: Test rotated angle method
    r⃗ = rand(3); Û = rand(3, 3)
    _, θ, φ = Hamster.transform_to_spherical(Û*r⃗)
    @test (θ, φ) == Hamster.get_rotated_angles(Û, r⃗)
end