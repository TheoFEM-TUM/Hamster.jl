const hamster_path = @__DIR__

# Precompiled at the end with `hamster --precompile` (see hamster_precompile.jl)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

const usage = """
Usage: hamster --install [--add_path yes|no] [--bashrc FILE] [--add_test_exec] [--test] [--gpu]
       (or: julia hamster_install.jl [options])
  --add_path yes|no   Add the Hamster folder to PATH in the bashrc (default: yes)
  --bashrc FILE       File to add PATH to (default: ~/.bashrc)
  --add_test_exec     Create a hamster_test executable to run the test suite
  --test              Run the test suite after the installation
  --gpu               Also install CUDA.jl and CUDSS.jl for GPU runs
"""

function parse_install_args(args)
    options = Dict{String, Any}("add_path" => "yes", "bashrc" => "default", "add_test_exec" => false, "test" => false, "gpu" => false)
    i = 1
    while i ≤ length(args)
        arg = args[i]
        if arg ∈ ("--add_path", "--bashrc", "--exec_name")
            i < length(args) || error("$arg requires a value.\n$usage")
            if arg == "--exec_name"
                @warn "--exec_name is no longer supported: the executable is always called hamster."
            else
                options[arg[3:end]] = args[i+1]
            end
            i += 2
        elseif arg ∈ ("--add_test_exec", "--test", "--gpu")
            options[arg[3:end]] = true
            i += 1
        elseif arg ∈ ("-h", "--help")
            print(usage)
            exit(0)
        else
            error("Unknown option $arg.\n$usage")
        end
    end
    options["add_path"] ∈ ("yes", "no") || error("--add_path must be yes or no.\n$usage")
    options["bashrc"] == "default" && (options["bashrc"] = joinpath(homedir(), ".bashrc"))
    return options
end

options = parse_install_args(ARGS)

println("""

Welcome to
===========================================================================
||   _   _                             _                         _   _   ||
||  | | | |   __ _   _ __ ___    ___  | |_    ___   _ __        (_) | |  ||
||  | |_| |  / _` | | '_ ` _ \\  / __| | __|  / _ \\ | '__|       | | | |  ||
||  |  _  | | (_| | | | | | | | \\__ \\ | |_  |  __/ | |     _    | | | |  ||
||  |_| |_|  \\__,_| |_| |_| |_| |___/  \\__|  \\___| |_|    (_)  _/ | |_|  ||
||                                                            |__/       ||
===========================================================================
Hamiltonian-learning
    Approach for Multiscale Simulations
        using a Transferable and Efficient Representation.

""")

# Loaded after the options, so that --help also works without Pkg in the load path (e.g., in Pkg.test)
using Pkg

# The user's default environment: JULIA_PROJECT if set (e.g., by the Julia modules on JSC systems), else @v#.#
const default_env = dirname(Base.active_project())

# Make Hamster available in the default environment (e.g., `using Hamster` in the REPL)
Pkg.activate(default_env)
Pkg.develop(PackageSpec(path=hamster_path))

println("Installing the dependencies of Hamster.jl...")
Pkg.activate(hamster_path)
Pkg.instantiate()

chmod(joinpath(hamster_path, "hamster"), 0o755)

if options["gpu"]
    println("Installing the GPU dependencies (CUDA.jl, CUDSS.jl) into $default_env...")
    Pkg.activate(default_env)
    Pkg.add(["CUDA", "CUDSS"])
end

println("Precompiling Hamster.jl...")
try
    run(`$(joinpath(hamster_path, "hamster")) --precompile`)
catch e
    println("Hamster was not installed successfully.")
    rethrow(e)
end

if options["add_test_exec"]
    test_exec_file = joinpath(hamster_path, "hamster_test")
    println("Generating the hamster_test executable...")
    write(test_exec_file, """
    #!/bin/bash
    HAMSTER_DIR="\$(dirname "\$(readlink -f "\${BASH_SOURCE[0]}")")"
    exec julia --project="\$HAMSTER_DIR" "\$HAMSTER_DIR/test/runtests.jl" "\$@"
    """)
    chmod(test_exec_file, 0o755)
end

if options["add_path"] == "yes"
    path_line = "export PATH=\"\$PATH:$hamster_path\""
    bashrc = options["bashrc"]
    if isfile(bashrc) && occursin(path_line, read(bashrc, String))
        println("Hamster.jl is already in PATH ($bashrc).")
    else
        println("Adding Hamster.jl to PATH ($bashrc)...")
        open(bashrc, "a") do io
            println(io, "\n# Add Hamster.jl to PATH.")
            println(io, path_line)
        end
    end
end

options["test"] && Pkg.test("Hamster")
println("Hamster was configured successfully. Run `hamster --help` to get started.")
