using Hamster, LinearAlgebra, BenchmarkTools, SparseArrays, OhMyThreads, Distributed, CubicSplines
using StaticArrays, HDF5, PyPlot
pygui(true)

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 18
rcParams["font.family"] = "serif"

x1 = SVector{8}(rand(8))

x2 = sparse(hcat([[SVector{8}(rand(8)) for i in 1:8] for j in 1:8]...))

Hamster.exp_sim.(x1, x2)

conf = get_config()

Hamster.get_sc_poscar(conf)

set_value!(conf, "poscar", "Supercell", "POSCAR")

t1 = [SVector{8}(rand(8)) for i in 1:100]
t2 = [SVector{8}(rand(8)) for i in 1:100]

reduce(vcat, [t1, t2])

Es = h5read("/home/martin/sshfs/juwels/hamster_test/ml_data_12_kernel_100/eigenval.h5", "eigenvalues")

path = "/home/martin/sshfs/juwels/hamster_test"

function read_loss_from_slurm(file, key)
    lines = Hamster.open_and_read(file)
    lines = Hamster.split_lines(lines)
    Ls = Float64[]
    for line in lines
        if key in line && "Iteration:" in line
            if key == "Val"
                push!(Ls, parse(Float64, line[8]))
            elseif key == "Batch"
                @show line[12]
                push!(Ls, parse(Float64, line[12]))
            end
        end
    end
    return Ls
end

loss_1 = read_loss_from_slurm(joinpath(path, "ml_data_12_kernel_50", "slurm-11069566.out"), "Val")
loss_2 = read_loss_from_slurm(joinpath(path, "ml_data_12_kernel_100", "slurm-11069568.out"), "Val")

plt.plot(loss_1, label="size = 50")
plt.plot(loss_2, label="size = 100")
plt.legend()
plt.xlabel("Iteration [-]"); plt.ylabel("Validation loss (eV)")