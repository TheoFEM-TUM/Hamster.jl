const hamster_path = @__DIR__

const usage = """
Usage: hamster --update [--no-pull]
       (or: julia hamster_update.jl [options])
  --no-pull   Only update the dependencies, do not pull the latest code with git
"""

pull = true
for arg in ARGS
    if arg == "--no-pull"
        global pull = false
    elseif arg ∈ ("-h", "--help")
        print(usage)
        exit(0)
    else
        error("Unknown option $arg.\n$usage")
    end
end

# Loaded after the options, so that --help also works without Pkg in the load path (e.g., in Pkg.test)
using Pkg

if pull
    is_git_repo = try
        success(pipeline(`git -C $hamster_path rev-parse --git-dir`, stdout=devnull, stderr=devnull))
    catch
        false
    end
    if is_git_repo
        println("Pulling the latest version of Hamster.jl...")
        try
            run(`git -C $hamster_path pull --ff-only`)
        catch e
            println("Pulling with git failed (e.g., local changes or a diverged branch). Resolve this and update again, " *
                    "or update only the dependencies with `hamster --update --no-pull`.")
            rethrow(e)
        end
    else
        println("$hamster_path is not a git repository, only the dependencies are updated.")
    end
end

try
    println("Starting Hamster update...")
    Pkg.activate(hamster_path)
    Pkg.update()
    println("Precompiling Hamster.jl...")
    run(`$(joinpath(hamster_path, "hamster")) --precompile`)
catch e
    println("There was an error while updating Hamster.")
    rethrow(e)
end
println("Hamster was updated successfully.")
