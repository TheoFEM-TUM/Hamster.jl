using Random: Xoshiro

# Two-band model on a periodic n × n × n cubic lattice: on-site energies ±Δ with disorder and nearest-neighbor hoppings with
# random phases, so that the spectrum has a gap around zero and no exact degeneracies. With `kramers`, every level is
# doubly degenerate (H ⊗ 1₂).
function chebyshev_test_matrix(n; Δ=1.5, t=0.5, kramers=false, seed=1)
    rng = Xoshiro(seed)
    site(x, y, z) = mod(x, n) + n * mod(y, n) + n^2 * mod(z, n)
    Is = Int[]; Js = Int[]; Vs = ComplexF64[]
    for x in 0:n-1, y in 0:n-1, z in 0:n-1, b in 1:2
        i = 2site(x, y, z) + b
        push!(Is, i); push!(Js, i); push!(Vs, (b == 1 ? -Δ : Δ) + 0.2 * randn(rng))
        for (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)], b2 in 1:2
            j = 2site(x + dx, y + dy, z + dz) + b2
            h = (b == b2 ? t : 0.3t) * cis(2π * rand(rng))
            append!(Is, [i, j]); append!(Js, [j, i]); append!(Vs, [h, conj(h)])
        end
    end
    H = sparse(Is, Js, Vs, 2n^3, 2n^3)
    return kramers ? kron(H, sparse(I, 2, 2)) : H
end

closest_eigenvalues(E, Neig, target) = sort(E[sortperm(abs.(E .- target))[1:Neig]])
