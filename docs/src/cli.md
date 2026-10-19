# [Command-Line Interface](@id man-cli)

The `hamster` executable in the `Hamster.jl` folder runs calculations and installs, updates and precompiles Hamster. It finds its own folder at runtime, so it can also be called through a symlink or from `PATH`.

```bash
hamster [-t THREADS] [--conf FILE] [--TAG VALUE ...]   # run a calculation
hamster --install [OPTIONS]                           # install the dependencies and precompile Hamster
hamster --update [--no-pull]                          # pull the latest code, update the dependencies, precompile
hamster --precompile                                  # precompile Hamster (and its GPU code on a GPU node)
hamster --version                                     # version of Hamster (and the git commit) and Julia
hamster --help                                        # overview of all commands and options
```

Only one command can be given per call (combining, e.g., `--update` and `--precompile` gives an error). Commands are recognized anywhere in the arguments, all other arguments are options of the command. `hamster --install --help` and `hamster --update --help` show the options of these commands.

## Running a calculation

```bash
[mpiexecjl -n NRANKS / srun] hamster [-t THREADS] [--conf FILE] [--TAG VALUE ...]
```

- `-t`, `--threads`: Number of Julia threads. Defaults to `SLURM_CPUS_PER_TASK` or `JULIA_NUM_THREADS`, if set.
- `--conf`: [Config file](@ref man-config) (default: `hconf` in the current directory).
- `--TAG VALUE`: Sets any config tag, e.g., `--verbosity 2`. Values given on the command line override the config file.

With MPI, start one `hamster` per rank with `mpiexecjl` or `srun`.

## Installation

```bash
hamster --install [--add_path yes/no] [--bashrc FILE] [--add_test_exec] [--test] [--gpu]
```

Installs the dependencies of Hamster, makes Hamster available in your default Julia environment (e.g., `using Hamster` in the REPL) and precompiles it. The default environment is the one `JULIA_PROJECT` points to, if set (as by the Julia modules on JSC systems), else `~/.julia/environments/v1.x`.

- `--add_path`: `yes` or `no` (default: `yes`). Adds the `Hamster.jl` folder to `PATH` in your `.bashrc` (only once, also if the installation is repeated).
- `--bashrc`: File to add `PATH` to (default: `~/.bashrc`).
- `--add_test_exec`: Also creates a `hamster_test` executable that runs the test suite.
- `--test`: Runs the test suite after the installation (up to 10 minutes).
- `--gpu`: Also installs [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) for GPU runs. The GPU code is precompiled only if a GPU is available. Otherwise (e.g., on the login node of a cluster), run `hamster --precompile` once on a GPU node.

`julia hamster_install.jl [options]` does the same as `hamster --install`. If you installed Hamster before the `hamster` executable became part of the repository, delete the generated `hamster` file in the `Hamster.jl` folder before pulling the new version (git refuses to overwrite it).

## Update

```bash
hamster --update [--no-pull]
```

Pulls the latest code with `git pull --ff-only`, updates the dependencies and precompiles. If the pull fails (e.g., because of local changes or a diverged branch), the update stops; resolve this and update again. With `--no-pull`, only the dependencies are updated, e.g., if you manage the code with git yourself. If the `Hamster.jl` folder is not a git repository, only the dependencies are updated.

## Precompilation

```bash
hamster --precompile
```

Precompiles Hamster and, if CUDA.jl is installed and a GPU is available, its GPU code. `--install` and `--update` do this automatically. Run it yourself on a GPU node if Hamster was installed or updated on a machine without GPU (e.g., the login node of a cluster): GPU runs then start in a few seconds instead of compiling the GPU code at every start.

## Version

```bash
hamster --version
```

Shows the version of Hamster, the git branch and commit (marked as modified if there are local changes), the path of the `Hamster.jl` folder and the Julia version.
