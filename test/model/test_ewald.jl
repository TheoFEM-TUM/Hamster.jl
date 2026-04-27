function brute_force_periodic_coulomb(
    pos::Matrix{Float64},
    q::Vector{Float64};
    box::Matrix{Float64},
    nimg::Int=3,
    subtract_mean_potential::Bool=false
)
    N = size(pos, 2)

    @assert size(pos, 1) == 3 "positions must be 3×N"
    @assert size(box) == (3, 3) "box must be 3×3"
    @assert length(q) == N "charges must have length N"

    phi = zeros(Float64, N)
    E = 0.0

    # image vectors n = (nx, ny, nz), with shift = box * n
    for i in 1:N
        ri = @view pos[:, i]

        for j in 1:N
            rj = @view pos[:, j]
            qj = q[j]

            for nx in -nimg:nimg, ny in -nimg:nimg, nz in -nimg:nimg
                # skip exact self interaction in the home cell
                if i == j && nx == 0 && ny == 0 && nz == 0
                    continue
                end

                shift = box * [nx, ny, nz]
                dr = ri - (rj + shift)
                r = norm(dr)

                phi[i] += ke * qj / r
            end
        end
    end

    # total energy = 1/2 sum_i q_i phi_i
    E = 0.5 * sum(q .* phi)

    if subtract_mean_potential
        phi .-= mean(phi)
    end

    return (
        potentials = phi,
        energy = E,
        nimg = nimg,
    )
end

@testset "Ewald" begin
    
end