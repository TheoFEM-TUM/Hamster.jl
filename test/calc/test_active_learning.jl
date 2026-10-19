@testset "Active learning" begin
    path = joinpath(@__DIR__, "test_files")
    function al_config(selection)
        conf = get_config(filename = joinpath(path, "hconf_md"))
        set_value!(conf, "poscar", joinpath(path, "POSCAR_md"))
        set_value!(conf, "poscar", "Supercell", joinpath(path, "POSCAR_md"))
        set_value!(conf, "xdatcar", "Supercell", joinpath(path, "XDATCAR"))
        set_value!(conf, "rllm_file", joinpath(path, "rllm.dat"))
        set_value!(conf, "train_data", "Optimizer", joinpath(path, "eigenval.h5"))
        set_value!(conf, "val_data", "Optimizer", joinpath(path, "eigenval.h5"))
        set_value!(conf, "val_mode", "Optimizer", "md")
        set_value!(conf, "init_params", joinpath(path, "params.dat"))
        set_value!(conf, "niter", "Optimizer", 2)
        set_value!(conf, "seed", 1)
        for (key, value) in ("nrounds" => 3, "nadd" => 2, "nval" => 2, "npoints_add" => 5, "selection" => selection)
            set_value!(conf, key, "ActiveLearning", value)
        end
        return conf
    end
    @test Hamster.decide_which_task_to_perform(al_config("coverage")) == Val{:active_learning}()

    for selection in ("coverage", "random")
        mktempdir() do dir
            cd(dir) do
                prof = Hamster.main(comm, al_config(selection), rank=rank)
                @test all(isfinite, prof.L_val) && all(prof.L_val .> 0)
                @test isfile("params.dat") && isfile("ml_params.dat") && isfile("al/round_3/params.dat")
                history = h5read("al/history.h5", "Ntrain")
                @test history == [2, 4, 6]
                val = h5read("al/history.h5", "val_config_inds")
                trains = [h5read("al/history.h5", "train_config_inds/round_$r") for r in 1:3]
                @test length(val) == 2 && all(isempty(intersect(t, val)) for t in trains)
                @test issubset(trains[1], trains[2]) && issubset(trains[2], trains[3])
                @test all(i -> 1 ≤ i ≤ 10, vcat(trains..., val))
                points = h5read("al/history.h5", "Npoints")
                @test issorted(points) && points[end] ≤ points[1] + 2 * 5
            end
        end
    end

    # Farthest-point extension: starts from the existing points, stops below the similarity threshold
    points = [SVector(0.0, 0.0)]
    candidates = [SVector(0.0, 0.01), SVector(1.0, 0.0), SVector(0.0, 2.0), SVector(0.0, 2.01)]
    new = Hamster.farthest_point_extension(candidates, points, 3, σ=0.1, add_below=0.5)
    @test new == [SVector(0.0, 2.01), SVector(1.0, 0.0)]
    @test isempty(Hamster.farthest_point_extension(candidates, candidates, 3, σ=0.1))
end
