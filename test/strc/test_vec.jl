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
