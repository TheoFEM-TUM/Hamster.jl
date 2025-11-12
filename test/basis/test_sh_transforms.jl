# Order: (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
import Hamster.dxy, Hamster.dyz, Hamster.dz2, Hamster.dxz, Hamster.dx2_y2
import Hamster.fdxy, Hamster.fdyz, Hamster.fdz2, Hamster.fdxz, Hamster.fdx2_y2

function get_coefficients(baseorb, θ, φ)
    return [fdxy(baseorb, θ, φ), fdyz(baseorb, θ, φ), fdz2(baseorb, θ, φ), fdxz(baseorb, θ, φ), fdx2_y2(baseorb, θ, φ)]
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