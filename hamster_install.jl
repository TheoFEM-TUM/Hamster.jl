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

using Pkg

# Add and import the `ArgParse` package
Pkg.add("ArgParse")
using ArgParse

hamster_path = string(@__DIR__)

Pkg.develop(PackageSpec(path=hamster_path))
Pkg.activate(hamster_path)
Pkg.update()
Pkg.resolve()
Pkg.instantiate()

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--exec_name"
            help = "defines the name of the Hamster executable alias"
            arg_type = String
            default = "hamster"
        "--bashrc"
            help = "sets the path where your executable is located"
            arg_type = String
            default = "default"
        "--add_path"
            help = "decides whether the Hamster package is added to path"
            arg_type = String
            default = "yes"
        "--add_test_exec"
            help = "if true, an executable to run individual test sets is created"
            action = :store_true
    end
    args :: Dict{String, Union{String, Bool}} = parse_args(s)
    return args
end

#Check if Hamster can be imported
try
    cd("..")
    using Hamster
    cd(hamster_path)
catch e
    println("Hamster was not installed successfully.")
    rethrow(e)
end

# Generate Hamster executable
args = parse_commandline()
exec_name = args["exec_name"]
exec_file = joinpath(hamster_path, exec_name)
if !isfile(exec_file)
    println("Generating Hamster.jl executable...")
    open(exec_file, "w+") do file
        println(file, "#!/bin/bash")
        println(file, "julia --project=$hamster_path $hamster_path/hamster_main.jl \"\$@\"")
    end
    println("")
    run(`chmod +x $exec_file`)
end

# Generate Hamster test executable
test_exec_file = joinpath(hamster_path, exec_name*"_test")
if !isfile(test_exec_file) && args["add_test_exec"]
    println("Generating Hamster.jl test executable...")
    open(test_exec_file, "w+") do file
        println(file, "#!/bin/bash")
        println(file, "julia --project=$hamster_path $hamster_path/test/runtests.jl \"\$@\"")
    end
    println("")
    run(`chmod +x $test_exec_file`)
end

# Add Hamster executable to path
if args["add_path"] == "yes"
    println("Adding Hamster.jl to PATH...")
    bashrc_path = args["bashrc"] == "default" ? joinpath(ENV["HOME"], ".bashrc") : args["bashrc"]
    open(bashrc_path, "a") do bashrc_file
        println(bashrc_file, "")
        println(bashrc_file, "# Add Hamster.jl to PATH.")
        println(bashrc_file, "export PATH=\"\$PATH:$hamster_path\"")
    end
    println("")
end

Pkg.test("Hamster")
println("Hamster was configured successfully.")