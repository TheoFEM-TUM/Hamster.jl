# Symmetry of the ML correction with enforce_symmetry_group = O_h, cubic CsPbBr3 (s, p orbitals).
# Orbitals: Pb s px py pz (1-4), Cs s (5), Br1 px py pz (6-8, at a/2 x), Br2 (9-11, a/2 y), Br3 (12-14, a/2 z).

const symm_path = joinpath(@__DIR__, "test_files")

function symm_config(group; soc=false, poscar=joinpath(symm_path, "POSCAR_CsPbBr3"))
    conf = get_config(filename=joinpath(symm_path, "hconf_cspbbr3"))
    set_value!(conf, "poscar", poscar)
    set_value!(conf, "rllm_file", joinpath(symm_path, "rllm_cspbbr3.dat"))
    set_value!(conf, "init_params", "ML", "rand")
    set_value!(conf, "init_params", "zeros")
    set_value!(conf, "Npoints", "ML", 60); set_value!(conf, "Ncluster", "ML", 10)
    set_value!(conf, "enforce_symmetry_group", "ML", group)
    soc && (set_value!(conf, "soc", true); set_value!(conf, "init_params", "SOC", "ones"))
    set_value!(conf, "verbosity", 0)
    return conf
end

function symm_kernel(conf; seed=1)
    strc = Structure(conf); basis = Basis(strc, conf)
    model = TBModel([strc], [basis], comm, conf)
    Hamster.Random.seed!(seed)
    kernel = HamiltonianKernel([strc], [basis], model, comm, conf)
    return strc, basis, kernel
end

R_index(strc, v) = findfirst(c -> strc.Rs[:, c] ≈ v, axes(strc.Rs, 2))

# Bands of the ML Hamiltonian at k-points related by the cubic symmetry (permutations of the components)
function cubic_violation(strc, Hr)
    stars = ([0.5 0 0; 0 0.5 0; 0 0 0.5], [0.5 0.5 0; 0.5 0 0.5; 0 0.5 0.5], [0.3 0.1 0; 0 0.3 0.1; 0.1 0 0.3],
             [0.3 0.1 0.2; -0.3 0.1 0.2; 0.3 -0.1 -0.2])
    return maximum(stars) do ks
        Es = [eigvals(Hermitian(Matrix(H))) for H in get_hamiltonian(Hr, strc.Rs, Matrix(transpose(ks)))]
        maximum(maximum(abs, E - Es[1]) for E in Es)
    end
end

@testset "Symmetry O_h: cubic and inversion symmetry" begin
    conf = symm_config("O_h")
    strc, basis, kernel = symm_kernel(conf)
    Hr = get_hr(kernel, Hamster.Dense(), 1)
    R0 = R_index(strc, [0, 0, 0]); Rx = R_index(strc, [1, 0, 0]); Rz = R_index(strc, [0, 0, 1])
    H(i, j, R) = real(Hr[R][i, j])
    # Bonds Pb-Br along +x (R = 0) and -x (Br image at r - R): s-p opposite, p-p σ and π equal; along z the same values
    @test abs(H(1, 6, R0)) > 1e-3
    @test H(1, 6, R0) ≈ -H(1, 6, Rx) atol=1e-10
    @test H(2, 6, R0) ≈ H(2, 6, Rx) atol=1e-10
    @test H(3, 7, R0) ≈ H(3, 7, Rx) atol=1e-10
    @test H(1, 14, R0) ≈ H(1, 6, R0) atol=1e-10            # Pb s - Br3 pz (bond z) = Pb s - Br1 px (bond x)
    @test H(1, 7, R0) ≈ 0 atol=1e-10                       # p perpendicular to the bond: no s-p coupling
    # On-site elements between orbitals of different parity vanish at the centrosymmetric Pb site
    @test all(j -> abs(H(1, j, R0)) < 1e-10, 2:4)
    @test H(2, 2, R0) ≈ H(3, 3, R0) ≈ H(4, 4, R0)
    # Hermitian and cubic (up to ~1e-6 from the TB geometry tensor, which enters the environment descriptors)
    @test all(R -> Hr[R] ≈ Matrix(Hr[R_index(strc, -strc.Rs[:, R])])', axes(strc.Rs, 2))
    @test cubic_violation(strc, Hr) < 1e-4
    # Without the group the kernel breaks the symmetry (what this option fixes)
    _, _, kernel_old = symm_kernel(symm_config("none"))
    @test cubic_violation(strc, get_hr(kernel_old, Hamster.Dense(), 1)) > 1e-2
end

@testset "Symmetry O_h: Kramers degeneracy with SOC" begin
    # At the time-reversal invariant points (Γ, X, M, R) time reversal alone keeps the Kramers pairs degenerate (H(R)
    # is real, the ML correction spin-independent), with or without the group
    trim = [0 0 0; 0.5 0 0; 0.5 0.5 0; 0.5 0.5 0.5]
    for group in ("O_h", "none")
        conf = symm_config(group, soc=true)
        strc = Structure(conf); basis = Basis(strc, conf)
        Hamster.Random.seed!(1)
        ham = EffectiveHamiltonian([strc], [basis], comm, conf)
        Es = [eigvals(Hermitian(Matrix(H))) for H in get_hamiltonian(ham, 1, Matrix(transpose(trim)))]
        @test maximum(maximum(abs, E[2:2:end] .- E[1:2:end]) for E in Es) < 1e-8
    end
    # At generic k the double degeneracy needs inversion and time reversal: broken without the group
    ks = [0.13 0.31 0.41; 0.07 0.22 0.17; 0.29 0.05 0.36]
    for (group, degenerate) in (("O_h", true), ("none", false))
        conf = symm_config(group, soc=true)
        strc = Structure(conf); basis = Basis(strc, conf)
        Hamster.Random.seed!(1)
        ham = EffectiveHamiltonian([strc], [basis], comm, conf)
        Es = [eigvals(Hermitian(Matrix(H))) for H in get_hamiltonian(ham, 1, ks)]
        splitting = maximum(maximum(abs, E[2:2:end] .- E[1:2:end]) for E in Es)
        @test (splitting < 1e-8) == degenerate          # inversion and time reversal: exact
    end
end

@testset "Symmetry O_h: distorted structure and on-site elements" begin
    mktempdir() do dir
        # Pb displaced along z (polar, C4v): on-site Pb s-pz is allowed, s-px and s-py are not, px and py are equivalent
        poscar = joinpath(dir, "POSCAR_polar")
        lines = readlines(joinpath(symm_path, "POSCAR_CsPbBr3"))
        lines[9] = "  0.0000000000000000  0.0000000000000000  0.0300000000000000"
        write(poscar, join(lines, "\n") * "\n")
        conf = symm_config("O_h", poscar=poscar)
        strc, basis, kernel = symm_kernel(conf)
        P = Hamster.shell_projections(strc, Hamster.shell_rows(basis), Hamster.get_ml_rcut(conf))[(1, 1)]   # Pb p
        @test abs(P[3]) > 1e-3 && abs(P[1]) < 1e-10 && abs(P[2]) < 1e-10
        Hr = get_hr(kernel, Hamster.Dense(), 1)
        R0 = R_index(strc, [0, 0, 0])
        @test abs(real(Hr[R0][1, 4])) > 1e-6                   # allowed (and odd in the displacement)
        @test abs(real(Hr[R0][1, 2])) < 1e-10 && abs(real(Hr[R0][1, 3])) < 1e-10
        @test real(Hr[R0][2, 2]) ≈ real(Hr[R0][3, 3])
        # C4 around z: k-points related by (kx, ky) -> (-ky, kx) have the same bands
        ks = [0.3 0.1 0.2; -0.1 0.3 0.2]
        Es = [eigvals(Hermitian(Matrix(H))) for H in get_hamiltonian(Hr, strc.Rs, Matrix(transpose(ks)))]
        @test Es[1] ≈ Es[2] atol=1e-8
    end
end

@testset "Symmetry O_h: apply_distortion and apply_distance_distortion" begin
    # Cubic reference positions, actual Pb position (pos - dist) displaced along z: polar (C4v) with apply_distortion
    # (angles and distances of the actual positions) and with apply_distance_distortion (actual distances only)
    for (key, polar) in (("apply_distortion", true), ("apply_distance_distortion", false))
        conf = symm_config("O_h")
        set_value!(conf, key, "ML", true)
        strc_0 = Structure(conf)
        rs = reduce(hcat, [ion.pos for ion in strc_0.ions]); types = [ion.type for ion in strc_0.ions]
        δ = zeros(3, length(types)); δ[3, 1] = -0.15
        strc = Structure(strc_0.Rs, rs, δ, types, strc_0.lattice, conf)
        basis = Basis(strc, conf); model = TBModel([strc], [basis], comm, conf)
        Hamster.Random.seed!(1)
        kernel = HamiltonianKernel([strc], [basis], model, comm, conf)
        rows = Hamster.shell_rows(basis); rcut = Hamster.get_ml_rcut(conf)
        P = Hamster.shell_projections(strc, rows, rcut, apply_distortion=polar)[(1, 1)]                   # Pb p
        @test abs(P[1]) < 1e-10 && abs(P[2]) < 1e-10 && (abs(P[3]) > 1e-3) == polar
        @test all(abs.(Hamster.shell_projections(strc, rows, rcut)[(1, 1)]) .< 1e-10)                    # reference: cubic
        Hr = get_hr(kernel, Hamster.Dense(), 1)
        R0 = R_index(strc, [0, 0, 0])
        @test abs(real(Hr[R0][1, 2])) < 1e-10 && abs(real(Hr[R0][1, 3])) < 1e-10
        @test real(Hr[R0][2, 2]) ≈ real(Hr[R0][3, 3])
        @test all(R -> Hr[R] ≈ Matrix(Hr[R_index(strc, -strc.Rs[:, R])])', axes(strc.Rs, 2))
        ks = [0.3 0.1 0.2; -0.1 0.3 0.2]
        Es = [eigvals(Hermitian(Matrix(H))) for H in get_hamiltonian(Hr, strc.Rs, Matrix(transpose(ks)))]
        @test Es[1] ≈ Es[2] atol=1e-8
    end
end

@testset "Symmetry O_h: hybrid orbitals are their own orbit" begin
    # GaAs with sp3 hybrids only: one descriptor per element with weight 1, with the distance and angles of
    # get_tb_descriptor (the environment values differ: squared instead of linear environment)
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename=joinpath(path, "hconf"))
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas")); set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
    set_value!(conf, "verbosity", 0)
    strc = Structure(conf); basis = Basis(strc, conf); model = TBModel(strc, basis, conf)
    d0 = Hamster.get_tb_descriptor(model.hs, model.params, strc, basis, conf)
    d, sym = Hamster.get_tb_descriptor_and_symmetry(model.hs, model.params, strc, basis, conf, group="O_h")
    @test all(R -> findnz(d[R])[1:2] == findnz(d0[R])[1:2], eachindex(d))
    @test all(R -> all(x -> x[1][1:6] ≈ x[2][1:6], zip(nonzeros(d[R]), nonzeros(d0[R]))), eachindex(d))
    @test all(R -> all(==(1), nonzeros(sym.W[R])) && all(==(1), diff(sym.W[R].colptr)), eachindex(d))
    @test all(R -> all(k -> sym.X[R][rowvals(sym.W[R])[k]] == nonzeros(d[R])[k], 1:nnz(d[R])), eachindex(d))
end

@testset "Symmetry O_h: precomputed similarities and gradient" begin
    conf = symm_config("O_h")
    strc, basis, kernel = symm_kernel(conf)
    Hr = get_hr(kernel, Hamster.Dense(), 1)
    # The precomputed similarity matrix of the same kernel (all similarities kept) gives the same Hamiltonian
    kp = Hamster.HamiltonianKernelPrecalced(kernel, 0, [strc.system], conf, sim_tol=0.0)
    Hr_pre = get_hr(kp, Hamster.Dense(), 1)
    @test all(R -> Hr_pre[R] ≈ Hr[R], eachindex(Hr))
    # Gradient of L = Σ G ⊙ H (dL/dH = G) against finite differences
    Hamster.Random.seed!(3)
    G = [randn(size(H)) for H in Hr]
    L(params) = (Hamster.set_params!(kernel, params); sum(sum(real.(H) .* g) for (H, g) in zip(get_hr(kernel, Hamster.Dense(), 1), G)))
    kernel.update = true
    p0 = copy(kernel.params)
    grad = Hamster.get_model_gradient(kernel, [1], Regularization(conf), [G])
    for n in (1, 7, 30)
        e = zeros(length(p0)); e[n] = 1e-4
        @test grad[n] ≈ (L(p0 .+ e) - L(p0 .- e)) / 2e-4 rtol=1e-5 atol=1e-8
    end
    Hamster.set_params!(kernel, p0)
end

@testset "Symmetry O_h: parameter file" begin
    mktempdir() do dir
        conf = symm_config("O_h")
        _, _, kernel = symm_kernel(conf)
        file = joinpath(dir, "ml_params")
        Hamster.write_params(kernel, conf, filename=file)
        @test occursin("enforce_symmetry_group = O_h", read(file * ".dat", String))
        @test Hamster.read_ml_params(conf, filename=file)[1] ≈ kernel.params
        # An unknown header entry gives a clear error
        write(file * ".dat", replace(read(file * ".dat", String), "enforce_symmetry_group = O_h" => "enforce_symmetry_group = O_h\n  antisymmetrize = false"))
        err = try Hamster.read_ml_params(conf, filename=file); nothing catch e; e end
        @test err isa ErrorException && occursin("unknown header entry antisymmetrize", err.msg)
        # Overwriting the file it was read from (init_params = output, settings from its header) keeps the header
        ml = get_empty_config()
        set_value!(ml, "init_params", "ML", file * ".dat")
        Hamster.write_params(kernel, ml, filename=file)
        @test Hamster.read_header_value(file * ".dat", "enforce_symmetry_group") == "O_h"
        # The default of the tag comes from the parameter file; a model trained with the group cannot be evaluated without it
        @test Hamster.get_ml_enforce_symmetry_group(ml) == "O_h"
        @test_throws ErrorException Hamster.read_ml_params(symm_config("none"), filename=file)
    end
end

# Covariance of the real-space ML correction under the operations g of a point group (about the origin):
# H[a i, b j] = Σ_kl D_ki D_lj H[g a k, g b l], with the representation D of g on the orbitals of each atom.
function covariance_violation(strc, basis, Hr, g)
    A = strc.lattice; Rs = strc.Rs
    pos = [ion.pos for ion in strc.ions]
    orbs = basis.orbitals; Norb = length.(orbs); off = cumsum([0; Norb[1:end-1]])
    Ridx = Dict(Tuple(round.(Int, Rs[:, r])) => r for r in axes(Rs, 2))
    function locate(r)
        f = A \ r
        for a in eachindex(pos)
            n = f - A \ pos[a]
            norm(n - round.(n)) < 1e-6 && return a, round.(Int, n)
        end
        error("no atom at $r")
    end
    function elem(ra, i, rb, j)
        a, na = locate(ra); b, nb = locate(rb)
        r = get(Ridx, Tuple(na - nb), nothing)         # orbital j of H(R) at r_j - R
        isnothing(r) ? 0.0 : real(Hr[r][off[a] + i, off[b] + j])
    end
    rep(a) = Hamster.shell_representation([o.type for o in orbs[a]], g)
    worst = 0.0
    for a in eachindex(pos), r in axes(Rs, 2), b in eachindex(pos)
        rb = pos[b] - A * Rs[:, r]
        Da, Db = rep(a), rep(b)
        for i in 1:Norb[a], j in 1:Norb[b]
            rhs = sum(Da[k, i] * Db[l, j] * elem(g * pos[a], k, g * rb, l) for k in 1:Norb[a], l in 1:Norb[b])
            worst = max(worst, abs(real(Hr[r][off[a] + i, off[b] + j]) - rhs))
        end
    end
    return worst
end

function symmetry_test_kernel(dir, poscar, atoms, rcut, group)
    write(joinpath(dir, "POSCAR"), poscar)
    conf = get_empty_config()
    set_value!(conf, "poscar", joinpath(dir, "POSCAR")); set_value!(conf, "rcut", rcut); set_value!(conf, "verbosity", 0)
    set_value!(conf, "rllm_file", joinpath(dir, "rllm.dat")); set_value!(conf, "init_params", "ones")
    for (element, orbitals, alpha) in atoms
        set_value!(conf, "orbitals", element, orbitals); set_value!(conf, "alpha", element, alpha)
    end
    set_value!(conf, "init_params", "ML", "rand"); set_value!(conf, "Npoints", "ML", 40); set_value!(conf, "Ncluster", "ML", 8)
    set_value!(conf, "enforce_symmetry_group", "ML", group); set_value!(conf, "env_scale", "ML", 6.0)
    strc = Structure(conf); basis = Basis(strc, conf)
    model = TBModel([strc], [basis], comm, conf)
    Hamster.Random.seed!(1)
    kernel = HamiltonianKernel([strc], [basis], model, comm, conf)
    return strc, basis, get_hr(kernel, Hamster.Dense(), 1)
end

@testset "Symmetry O_h: d orbitals (SrTiO3, eg mixed by C3)" begin
    srtio3 = """
        SrTiO3
        3.905
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
        Sr Ti O
        1 1 3
        Direct
        0.0 0.0 0.0
        0.5 0.5 0.5
        0.5 0.5 0.0
        0.5 0.0 0.5
        0.0 0.5 0.5
        """
    atoms = (("Sr", "s", 36.0), ("Ti", "dxy dyz dxz dz2 dx2_y2", 19.0), ("O", "px py pz", 5.6))
    G = Hamster.point_group("O_h")
    mktempdir() do dir
        strc, basis, Hr = symmetry_test_kernel(dir, srtio3, atoms, 4.0, "O_h")
        @test maximum(abs, real(Hr[R_index(strc, [0, 0, 0])])) > 1e-3
        @test all(R -> Hr[R] ≈ Matrix(Hr[R_index(strc, -strc.Rs[:, R])])', axes(strc.Rs, 2))     # Hermitian
        # exact up to the geometry tensor (distance functions interpolated to ~1e-6), which enters the environment
        @test maximum(g -> covariance_violation(strc, basis, Hr, g), G[1:12]) < 1e-5
    end
    mktempdir() do dir
        strc, basis, Hr = symmetry_test_kernel(dir, srtio3, atoms, 4.0, "none")
        @test maximum(g -> covariance_violation(strc, basis, Hr, g), G[1:12]) > 1e-3      # broken without the group
    end
end

@testset "Symmetry D_6h: p orbitals (simple hexagonal, px/py mixed by C6)" begin
    hexagonal = """
        simple hexagonal
        1.0
        3.0 0.0 0.0
        -1.5 2.598076211353316 0.0
        0.0 0.0 3.6
        Ga
        1
        Direct
        0.0 0.0 0.0
        """
    atoms = (("Ga", "s px py pz", 14.3),)
    G = Hamster.point_group("D_6h")
    mktempdir() do dir
        strc, basis, Hr = symmetry_test_kernel(dir, hexagonal, atoms, 4.0, "D_6h")
        @test maximum(g -> covariance_violation(strc, basis, Hr, g), G) < 1e-8
    end
    mktempdir() do dir
        strc, basis, Hr = symmetry_test_kernel(dir, hexagonal, atoms, 4.0, "none")
        @test maximum(g -> covariance_violation(strc, basis, Hr, g), G) > 1e-3
    end
end

@testset "Symmetry: orientation of the lattice" begin
    a = 3.96
    c, s = cosd(30), sind(30)
    cubic = a * Matrix(1.0I, 3, 3)
    hexagonal = [3.0 -1.5 0; 0 3 * sqrt(3) / 2 0; 0 0 5.0]
    tetragonal = Matrix(Diagonal([a, a, 1.02a]))
    rotated = [c -s 0; s c 0; 0 0 1] * cubic
    @test Hamster.lattice_point_group_order(cubic) == 48 && Hamster.lattice_point_group_order(rotated) == 48
    @test Hamster.lattice_point_group_order(hexagonal) == 24 && Hamster.lattice_point_group_order(tetragonal) == 16
    @test Hamster.check_symmetry_orientation(cubic, "O_h", write_output=false) == (48, 48, 8)
    @test Hamster.check_symmetry_orientation(tetragonal, "O_h", write_output=false)[1] == 16
    @test (@test_logs Hamster.check_symmetry_orientation(hexagonal, "D_6h", write_output=false)) == (24, 24, 8)
    # misaligned or the wrong group: warning
    @test_logs (:warn, r"not aligned") Hamster.check_symmetry_orientation(rotated, "O_h", write_output=false)
    @test_logs (:warn, r"D_6h fits better") Hamster.check_symmetry_orientation(hexagonal, "O_h", write_output=false)
end
