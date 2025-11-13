# p-orbitals
import Hamster.px, Hamster.py, Hamster.pz
import Hamster.fpx, Hamster.fpy, Hamster.fpz

# Order: (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
import Hamster.dxy, Hamster.dyz, Hamster.dz2, Hamster.dxz, Hamster.dx2_y2
import Hamster.fdxy, Hamster.fdyz, Hamster.fdz2, Hamster.fdxz, Hamster.fdx2_y2

function get_coefficients(baseorb, θ, φ)
    return [fdxy(baseorb, θ, φ), fdyz(baseorb, θ, φ), fdz2(baseorb, θ, φ), fdxz(baseorb, θ, φ), fdx2_y2(baseorb, θ, φ)]
end

function get_analytic_d_orbital(baseorb, r0, θ, φ, n, α)
    Rllms = [Hamster.R1(α), Hamster.R2(α), Hamster.R3(α), Hamster.R4(α), Hamster.R5(α), Hamster.R6(α)]
    orbs = [dxy(), dyz(), dz2(), dxz(), dx2_y2()]
    coefs = get_coefficients(baseorb, θ, φ)
    @show coefs
    return r⃗ -> sum([coef*orb(r⃗ - r0) for (coef, orb) in zip(coefs, orbs)]) * Rllms[n](r⃗ - r0)
end

function get_tb_d_overlap(base1, base2, r0, θ₁, φ₁, θ₂, φ₂, ns, αs)
    r = norm(r0)
    orbconfig = Hamster.SymOrb()
    mode = Hamster.ConjugateMode()
    Cllms = [Hamster.Vddσ((2, 2)), Hamster.Vddπ((2, 2)), Hamster.Vddδ((2, 2))]
    Is = map(Cllms) do Cllm
        Cllm((base1, base2), orbconfig, mode, θ₁, φ₁, θ₂, φ₂)*Hamster.distance_dependence(Cllm, orbconfig, r, ns, αs)
    end
    return Is
end

function get_analytic_p_orbital(baseorb, r0, θ, φ, n, α)
    Rllms = [Hamster.R1(α), Hamster.R2(α), Hamster.R3(α), Hamster.R4(α), Hamster.R5(α), Hamster.R6(α)]
    orbs = [px(), py(), pz()]
    coefs = [fpx(θ, φ), fpy(θ, φ), fpz(θ, φ)]
    return r⃗ -> sum([coef*orb(r⃗ - r0) for (coef, orb) in zip(coefs, orbs)]) * Rllms[n](r⃗ - r0)
end

function get_tb_p_overlap(base1, base2, r0, θ₁, φ₁, θ₂, φ₂, ns, αs)
    r = norm(r0)
    orbconfig = Hamster.SymOrb()
    Cllms = [Hamster.Vppσ((1, 1)), Hamster.Vppπ((1, 1))]
    I = mapreduce(+, Cllms) do Cllm
        Cllm((base1, base2), orbconfig, Hamster.NormalMode(), θ₁, φ₁, θ₂, φ₂)*Hamster.distance_dependence(Cllm, orbconfig, r, ns, αs)
    end
    return I
end

@testset "SH transform d-orbitals" begin
    _, θ, φ = Hamster.transform_to_spherical([0., 0., 1.])
    _, θ′, φ′ = Hamster.transform_to_spherical(rand(3) .- 0.5)
    for (iorb, orb) in enumerate([dxy(), dyz(), dz2(), dxz(), dx2_y2()])
        coef = get_coefficients(orb, θ, φ)
        @test coef ≈ (i -> i == iorb ? 1.0 : 0.0).(1:length(coef))

        coef_i = get_coefficients(orb, θ′, φ′)
        for (jorb, orb_j) in enumerate([dxy(), dyz(), dz2(), dxz(), dx2_y2()])
            coef_j = get_coefficients(orb_j, θ′, φ′)
            @test coef_i ⋅ coef_j ≈ ifelse(iorb == jorb, 1, 0) atol=1e-15
        end
    end
end

@testset "p-orbitals integration" begin
    for i in 1:3
        r0 = 2 .* rand(3) .- 1
        axis1 = rand(3) .- 0.5
        axis2 = rand(3) .- 0.5

        _, θ1, φ1 = Hamster.transform_to_spherical(axis1)
        _, θ2, φ2 = Hamster.transform_to_spherical(axis2)

        
        orb1 = get_analytic_p_orbital(px(), zeros(3), θ1, φ1, 3, 11.)
        orb2 = get_analytic_p_orbital(px(), r0, θ2, φ2, 3, 11.)

        f(r) = orb1(r) * orb2(r)
        xmax = 20
        I_analytic, _ = hcubature(f, [-xmax, -xmax, -xmax], [xmax, xmax, xmax], maxevals=500000, initdiv=5, atol=1e-5)

        Û = Hamster.get_sk_transform_matrix(zeros(3), r0, axis1, "Rotation")
        θ₁, φ₁ = Hamster.get_rotated_angles(Û, axis1)
        θ₂, φ₂ = Hamster.get_rotated_angles(Û, axis2)
        I_tb = get_tb_p_overlap(px(), px(), r0, θ₁, φ₁, θ₂, φ₂, [3, 3], [11., 11.])
        @test I_analytic ≈ I_tb atol=1e-5
    end
end

@testset "d-orbitals integration" begin
    r0 = [1., 0.5, 0.] #2 .* rand(3) .- 1
    axis1 = [1., 0., 0.]#rand(3) .- 0.5
    axis2 = [1., 0., 0.]#rand(3) .- 0.5

    f1 = dyz()
    f2 = dxy()

    _, θ1, φ1 = Hamster.transform_to_spherical(axis1)
    _, θ2, φ2 = Hamster.transform_to_spherical(axis2)

    orb1 = get_analytic_d_orbital(f1, zeros(3), θ1, φ1, 3, 11.)
    orb2 = get_analytic_d_orbital(f2, r0, θ2, φ2, 3, 11.)

    f(r) = orb1(r) * orb2(r)
    I0, _ = hcubature(f, [-10, -10, -10], [15, 15, 15], maxevals=1000000, initdiv=5)
    @show I0

    Û = Hamster.get_sk_transform_matrix(zeros(3), r0, axis1, "Rotation")
    θ₁, φ₁ = Hamster.get_rotated_angles(Û, axis1)
    θ₂, φ₂ = Hamster.get_rotated_angles(Û, axis2)
    @show θ₁, φ₁, θ₂, φ₂
    Is = get_tb_d_overlap(f1, f2, r0, θ₁, φ₁, θ₂, φ₂, [3, 3], [11., 11.])
    @show Is
    @show sum(Is)
end