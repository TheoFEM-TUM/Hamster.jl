using PrecompileTools: @compile_workload, @setup_workload

# The precompile workload runs small but complete Hamster calculations (as the `hamster` executable does via `main`),
# such that the code paths of typical runs are compiled when the package is precompiled instead of at every start.
# The structures and parameters are in src/precompile; eigenvalue data and MD snapshots are generated on the fly.

function write_precompile_eigenval(file, nbands, kpoints; seed=1)
    rng = Random.MersenneTwister(seed)
    open(file, "w") do io
        println(io, "    2    2    1    1")
        println(io, "  0.2254527E+02  0.3995153E-09  0.3995153E-09  0.3995153E-09  0.5000000E-15")
        println(io, "  1.000000000000000E-004")
        println(io, "  CAR ")
        println(io, " precompile")
        println(io, "      8      $(size(kpoints, 2))     $nbands")
        for k in axes(kpoints, 2)
            println(io, " ")
            println(io, "  ", join(kpoints[:, k], "  "), "  1.0")
            for (j, E) in enumerate(sort(10 .* rand(rng, nbands) .- 8))
                println(io, "    $j  $E  1.000000")
            end
        end
    end
end

function write_precompile_md_data(dir, data)
    lines = readlines(joinpath(data, "SC_POSCAR"))
    natoms = sum(parse.(Int, split(lines[7])))
    positions = reduce(hcat, [parse.(Float64, split(lines[8 + i])[1:3]) for i in 1:natoms])
    rng = Random.MersenneTwister(2)
    h5open(joinpath(dir, "structures.h5"), "w") do f
        f["configs"] = cat([mod.(positions .+ 0.005 .* randn(rng, size(positions)), 1) for _ in 1:2]..., dims=3)
        f["lattice"] = [5.65 0 0; 0 5.65 0; 0 0 5.65]
    end
    h5open(joinpath(dir, "eigenval.h5"), "w") do f
        f["eigenvalues"] = sort(10 .* rand(rng, 32, 2, 2) .- 8, dims=1)
        f["kpoints"] = [0.0 0.25; 0.0 0.0; 0.0 0.25]
    end
end

const PRECOMPILE_ATOMS = """
begin atom Ga
    n = 3
    alpha = 9
    orbitals = sp3dr2 sp3dr2 sp3dr2 sp3dr2
end

begin atom As
    n = 3
    alpha = 13
    orbitals = sp3dr2 sp3dr2 sp3dr2 sp3dr2
    nnaxes = true
end
"""

function precompile_configs(dir, data)
    options(lines...) = "begin Options\n    verbosity = 0\n    load_rllm = true\n    rcut = 7\n    rllm_file = $(joinpath(data, "rllm.dat"))\n" *
                        join("    " .* collect(lines), "\n") * "\nend\n"
    block(name, lines...) = "\nbegin $name\n" * join("    " .* collect(lines), "\n") * "\nend\n"
    poscar = "poscar = $(joinpath(data, "POSCAR_gaas"))"
    params = "init_params = $(joinpath(data, "params.dat"))"
    eigenval = joinpath(dir, "EIGENVAL"); eigenval_soc = joinpath(dir, "EIGENVAL_soc")
    md = block("Supercell", "Nconf = 2", "Nconf_min = 1", "Nconf_max = 2", "xdatcar = $(joinpath(dir, "structures.h5"))", "poscar = $(joinpath(data, "SC_POSCAR"))")
    optimizer(train, lines...) = block("Optimizer", "niter = 2", "lr = 0.01", "loss = MAE", "train_data = $train", lines...)
    ml(lines...) = block("ML", "ncluster = 3", "npoints = 10", lines...)
    return [
        # Standard runs: dense, sparse with shift-invert, MD snapshots with ML
        options(poscar, params, "kpoints = $eigenval") * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true", "sp_diag = true", "neig = 2", "eig_target = 3.0") * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true", "kpoints = $(joinpath(dir, "eigenval.h5"))") * md * ml("init_params = rand") * PRECOMPILE_ATOMS,
        # Optimization: TB (dense, with validation), TB (sparse), ML with and without sim_mat (MD data), SOC
        options(poscar, params) * optimizer(eigenval, "val_data = $eigenval") * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true") * optimizer(eigenval) * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true") * md * optimizer(joinpath(dir, "eigenval.h5"), "train_mode = md", "update_tb = false") * ml() * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true") * md * optimizer(joinpath(dir, "eigenval.h5"), "train_mode = md", "update_tb = false") * ml("sim_mat = true") * PRECOMPILE_ATOMS,
        options(poscar, "init_params = $(joinpath(data, "params_soc.dat"))", "soc = true") * optimizer(eigenval_soc) * block("SOC", "update = true", "init_params = zeros") * PRECOMPILE_ATOMS,
    ]
end

@setup_workload begin
    data = joinpath(@__DIR__, "precompile")
    @compile_workload begin
        MPI.Initialized() || MPI.Init()
        comm = MPI.COMM_WORLD
        mktempdir() do dir
            kpoints = [0.0 0.25 0.5; 0.0 0.0 0.5; 0.0 0.25 0.5]
            write_precompile_eigenval(joinpath(dir, "EIGENVAL"), 8, kpoints)
            write_precompile_eigenval(joinpath(dir, "EIGENVAL_soc"), 16, kpoints, seed=2)
            write_precompile_md_data(dir, data)
            for (n, text) in enumerate(precompile_configs(dir, data))
                run_dir = mkpath(joinpath(dir, "run$n"))
                hconf = joinpath(run_dir, "hconf")
                write(hconf, text)
                cd(run_dir) do
                    redirect_stdout(devnull) do
                        main(comm, get_config(filename=hconf))
                    end
                end
            end
        end
    end
end
