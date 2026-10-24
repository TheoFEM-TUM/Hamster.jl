# COV_EXCL_START
using PrecompileTools: @compile_workload

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

# MD snapshots and eigenvalues in the universal format (one group per system), with SOC eigenvalues in eigenval_soc.h5
function write_precompile_md_data(dir, data)
    lines = readlines(joinpath(data, "SC_POSCAR"))
    counts = parse.(Int, split(lines[7]))
    natoms = sum(counts)
    positions = reduce(hcat, [parse.(Float64, split(lines[8 + i])[1:3]) for i in 1:natoms])
    atom_types = reduce(vcat, [fill(element_to_number(type), n) for (type, n) in zip(split(lines[6]), counts)])
    rng = Random.MersenneTwister(2)
    h5open(joinpath(dir, "structures.h5"), "w") do f
        g = create_group(f, "GaAs")
        g["configs"] = cat([mod.(positions .+ 0.005 .* randn(rng, size(positions)), 1) for _ in 1:3]..., dims=3)
        g["rs_atom"] = positions
        g["lattice"] = [5.65 0 0; 0 5.65 0; 0 0 5.65]
        g["atom_types"] = atom_types
    end
    for (file, nbands) in (("eigenval.h5", 32), ("eigenval_soc.h5", 64))
        h5open(joinpath(dir, file), "w") do f
            g = create_group(f, "GaAs")
            g["eigenvalues"] = sort(10 .* rand(rng, nbands, 2, 3) .- 8, dims=1)
            g["kpoints"] = repeat([0.0 0.25; 0.0 0.0; 0.0 0.25], 1, 1, 3)
            g["N_VBM"] = 16
            g["nelec"] = 32
        end
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

# The ML descriptor needs s and p orbitals (no hybrid orbitals)
const PRECOMPILE_ATOMS_ML = """
begin atom Ga
    n = 3
    alpha = 9
    orbitals = s px py pz
end

begin atom As
    n = 3
    alpha = 13
    orbitals = s px py pz
end
"""

function precompile_configs(dir, data)
    options(lines...) = "begin Options\n    verbosity = 0\n    load_rllm = true\n    rcut = 7\n    rllm_file = $(joinpath(data, "rllm.dat"))\n" *
                        join("    " .* collect(lines), "\n") * "\nend\n"
    block(name, lines...) = "\nbegin $name\n" * join("    " .* collect(lines), "\n") * "\nend\n"
    poscar = "poscar = $(joinpath(data, "POSCAR_gaas"))"
    params = "init_params = $(joinpath(data, "params.dat"))"
    eigenval = joinpath(dir, "EIGENVAL"); eigenval_h5 = joinpath(dir, "eigenval.h5"); eigenval_soc_h5 = joinpath(dir, "eigenval_soc.h5")
    md(lines...) = block("Supercell", "Nconf = 2", "Nconf_min = 1", "Nconf_max = 3", "xdatcar = $(joinpath(dir, "structures.h5"))", "poscar = $(joinpath(data, "SC_POSCAR"))", lines...)
    optimizer(train, lines...) = block("Optimizer", "niter = 2", "lr = 0.01", "loss = MAE", "train_data = $train", "train_mode = universal", lines...)
    ml(lines...) = block("ML", "ncluster = 3", "npoints = 10", "sample_strat = split", lines...)
    return [
        # Standard runs: dense, sparse with shift-invert, MD snapshots with ML
        options(poscar, params, "kpoints = $eigenval") * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true", "sp_diag = true", "neig = 2", "eig_target = 3.0") * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true", "kpoints = $eigenval_h5") * md() * ml("init_params = rand") * PRECOMPILE_ATOMS_ML,
        # Optimization (MD data): TB (dense, with validation), TB (sparse), ML with and without batched SimMats, SOC
        options(poscar, params) * md() * optimizer(eigenval_h5, "val_mode = universal", "val_data = $eigenval_h5", "val_ratio = 0.5") * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true") * md() * optimizer(eigenval_h5) * PRECOMPILE_ATOMS,
        options(poscar, params, "sp_mode = true") * md() * optimizer(eigenval_h5, "update_tb = false") * ml() * PRECOMPILE_ATOMS_ML,
        options(poscar, params, "sp_mode = true") * md("nbatch = 2") * optimizer(eigenval_h5, "update_tb = false") *
            ml("kernel_strat = Hamerly") * PRECOMPILE_ATOMS_ML,
        options(poscar, "init_params = $(joinpath(data, "params_soc.dat"))", "soc = true") * md() * optimizer(eigenval_soc_h5) *
            block("SOC", "update = true", "init_params = zeros") * PRECOMPILE_ATOMS,
    ]
end

# Standard runs on the GPU, as in large MD-based calculations (SOC, ML, sparse): Hamiltonians only (written to file),
# shift-invert and dense diagonalization. Run by the precompile workload of the CUDSS extension.
function precompile_gpu_configs(dir, data)
    options(lines...) = "begin Options\n    verbosity = 0\n    load_rllm = true\n    rcut = 7\n    device = gpu\n" *
                        "    rllm_file = $(joinpath(data, "rllm.dat"))\n    poscar = $(joinpath(data, "POSCAR_gaas"))\n" *
                        join("    " .* collect(lines), "\n") * "\nend\n"
    block(name, lines...) = "\nbegin $name\n" * join("    " .* collect(lines), "\n") * "\nend\n"
    md = block("Supercell", "Nconf = 2", "Nconf_min = 1", "Nconf_max = 2", "xdatcar = $(joinpath(dir, "structures.h5"))", "poscar = $(joinpath(data, "SC_POSCAR"))")
    rest = md * block("ML", "ncluster = 3", "npoints = 10", "sample_strat = split", "init_params = rand") * PRECOMPILE_ATOMS_ML
    soc = ("init_params = $(joinpath(data, "params_soc.dat"))", "soc = true")
    return [
        options(soc..., "sp_mode = true", "skip_diag = true", "write_hk = true") * rest,
        options(soc..., "sp_mode = true", "sp_diag = true", "neig = 2", "eig_target = 3.0") * rest,
        options("init_params = $(joinpath(data, "params.dat"))", "sp_mode = true", "sp_diag = true", "neig = 2", "eig_target = 3.0") * rest,
        options(soc...) * rest,
    ]
end

# Runs `main` for each config returned by `make_configs(dir, data)` in a temporary directory
function run_precompile_workload(make_configs)
    data = joinpath(@__DIR__, "precompile")
    # Do not join the job of the MPI launcher
    for key in collect(keys(ENV))
        any(startswith(key, prefix) for prefix in ("PMI_", "PMIX_", "OMPI_", "HYDRA_", "MPI_LOCALRANKID", "MPI_LOCALNRANKS")) &&
            delete!(ENV, key)
    end
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    mktempdir() do dir
        kpoints = [0.0 0.25 0.5; 0.0 0.0 0.5; 0.0 0.25 0.5]
        write_precompile_eigenval(joinpath(dir, "EIGENVAL"), 8, kpoints)
        write_precompile_md_data(dir, data)
        for (n, text) in enumerate(make_configs(dir, data))
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

@compile_workload begin
    run_precompile_workload(precompile_configs)
end
# COV_EXCL_STOP
