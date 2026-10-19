# Precompiles Hamster and, if CUDA.jl is installed and a GPU is available, its GPU code. Run by `hamster --precompile`
# with the Hamster folder as active project: then the GPU extensions are compiled together with their precompile
# workload (with the default environment as active project, the workload of the CUDSS extension is skipped).
using Hamster

if isnothing(Base.find_package("CUDA"))
    println("Hamster.jl is precompiled.")
else
    using CUDA
    if Base.invokelatest(getfield(CUDA, :functional))
        if !isnothing(Base.find_package("CUDSS"))
            using CUDSS
        end
        println("Hamster.jl and its GPU code are precompiled.")
    else
        println("Hamster.jl is precompiled. No functional GPU found: run `hamster --precompile` on a GPU node to " *
                "precompile the GPU code (otherwise, it is compiled when Hamster runs on a GPU node for the first time).")
    end
end
