angular_parts = [Hamster.s(), Hamster.px(), Hamster.py(), Hamster.pz(), Hamster.dyz(), Hamster.dxz(), Hamster.dxy(), Hamster.dz2()]
radial_parts = [Hamster.R1(11.), Hamster.R2(11.), Hamster.R3(11.), Hamster.R4(11.), Hamster.R5(11.), Hamster.R6(11.)]

function not_nan_at_zero(angular_parts, radial_parts)
    r0 = zeros(Float64, 3)
    for Yₗ in angular_parts, Rᵣ in radial_parts
        if isnan((Yₗ(r0)*Rᵣ(r0))); return false; end
    end
    true
end

function are_normalized(angular_parts, Rᵣ)
    for Yₗ in angular_parts, Rᵣ in radial_parts
        if Yₗ.l ≤ Rᵣ.n - 1
            f(r⃗) = (Yₗ(r⃗)*Rᵣ(r⃗))*(Yₗ(r⃗)*Rᵣ(r⃗))
            I0, _ = hcubature(f, [-20, -20, -20], [20, 20, 20], maxevals=1000000,
                                initdiv=5)
            ε = 1e-4
            if abs(I0 - 1.) ≥ ε; @show I0, typeof(Rᵣ), typeof(Yₗ); return false; end
        end
    end
    true
end

function are_orthogonal(angular_parts, Rᵣ)
    for Y1ₗ in angular_parts, Y2ₗ in angular_parts
        if typeof(Y1ₗ) ≠ typeof(Y2ₗ)
            f(r⃗) = (Y1ₗ(r⃗)*Rᵣ(r⃗))*(Y2ₗ(r⃗)*Rᵣ(r⃗))
            I0, _ = hcubature(f, [-20, -20, -20], [20, 20, 20], maxevals=1000000,
                                initdiv=5)
            ε = 1e-5
            if abs(I0) ≥ ε; return false; end
        end
    end
    true
end

@testset "Spherical Harmonics" begin
    @test not_nan_at_zero(angular_parts, radial_parts)
    @test are_normalized(angular_parts, radial_parts)
    @test are_orthogonal(angular_parts[1:end-2], radial_parts[3])
end