using Pkg

hamster_path = string(@__DIR__)

#Try to update Hamster
try
    println("Starting Hamster update...")
    Pkg.activate(hamster_path)
    Pkg.update()
    Pkg.resolve()
catch e
    println("There was an error while updating Hamster.")
    rethrow(e)
end