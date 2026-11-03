# Runs of test/calc/test_mpi.jl with two MPI ranks: julia mpi_run.jl <case> <folder>
using Hamster, MPI, HDF5
include(joinpath(@__DIR__, "..", "universal_helpers.jl"))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)
case, dir = ARGS
path = @__DIR__

if case == "abort"
    # Rank 0 waits in a collective call while rank 1 fails
    Hamster.abort_on_error(comm) do
        rank == 1 && error("test error on rank 1")
        MPI.Barrier(comm)
    end
elseif case == "runs"
    # MD standard run, one structure per rank
    cd(joinpath(dir, "md")) do
        conf = get_config(filename = joinpath(path, "hconf_std_md"))
        set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
        set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
        set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
        set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
        set_value!(conf, "kpoints", joinpath(path, "eigenval.h5"))
        set_value!(conf, "init_params", joinpath(path, "params.dat"))
        set_value!(conf, "write_hk", true)
        set_value!(conf, "write_hr", true)
        Hamster.main(comm, conf, rank=rank, nranks=nranks)
    end
    # Standard run with two systems, one system per rank
    cd(joinpath(dir, "universal")) do
        strc_file = joinpath(dir, "universal", "strc_file.h5")
        rank == 0 && write_universal_strc_file(strc_file)
        MPI.Barrier(comm)
        conf = universal_standard_config(strc_file)
        Hamster.TEMP_DIR[] = "run_tmp"
        rank == 0 && mkdir("run_tmp")
        MPI.Barrier(comm)
        Hamster.run_calculation(Val{:standard}(), comm, conf, rank=rank, nranks=nranks, write_output=false)
        rank == 0 && Hamster.collapse_eigenvalue_files(conf)
    end
end
MPI.Finalize()
