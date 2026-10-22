"""
    run_calculation(::Val{:md}, comm, conf; rank=0, nranks=1, write_output=true)

Molecular dynamics with energies and forces from the Hamster Hamiltonian (switched on by the `MD` block). The `task` of
the MD block selects:

1. `fit`: learn the nuclear term (`MDRepulsive`) from DFT energies and forces (`train_data`, validated on `val_data`)
   and write it to `model`. With a `HyperOpt` block, its hyperparameters (e.g. `MD_rep_rcut`, `MD_lambda`) are optimized
   for the validation loss; the electronic energies and forces are computed once and cached (`cache`).
2. `eval`: energies and forces of `train_data` with a fitted model and their errors.
3. `run`: NVT molecular dynamics of the reference structure (`poscar`, repeated `supercell` times), written to `output`,
   with the analysis (radial distribution functions, vibrational density of states, MSD) at the end.

Energies and forces: the Mermin free energy of the electrons (TB, Wolf onsite corrections, SOC) at the electronic
temperature `kt`, by exact diagonalization (k-points) or the linear-scaling kernel polynomial method (Γ, CPU or GPU),
plus the nuclear term.
"""
function run_calculation(::Val{:md}, comm, conf::Config; rank=0, nranks=1, write_output=true, verbosity=get_verbosity(conf))
    task = lowercase(get_md_task(conf))
    if task == "fit"
        return run_md_fit(comm, conf; rank=rank, nranks=nranks, verbosity=verbosity, write_output=write_output)
    elseif task == "eval"
        return run_md_eval(comm, conf; rank=rank, nranks=nranks, verbosity=verbosity)
    elseif task == "run"
        return run_md(comm, conf; rank=rank, nranks=nranks, verbosity=verbosity)
    end
    error("Unknown task = $task in the MD block (fit, eval or run).")
end

save(::MDState, rank=0; filename="hamster_out.h5") = nothing
save(::Nothing, rank=0; filename="hamster_out.h5") = nothing
