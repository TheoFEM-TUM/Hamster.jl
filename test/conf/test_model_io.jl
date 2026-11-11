using Hamster, Test, MPI, HDF5, LinearAlgebra

# This file can also run directly under mpiexecjl with --project=.
MPI.Initialized() || MPI.Init()

function model_io_config()
    conf = get_empty_config()
    fixtures = joinpath(@__DIR__, "..", "model", "test_files")
    for (key, value) in (("poscar", joinpath(fixtures, "POSCAR_gaas")),
                         ("rllm_file", joinpath(fixtures, "rllm_true.dat")),
                         ("init_params", joinpath(fixtures, "params.dat")),
                         ("load_rllm", true), ("rcut", 5), ("verbosity", 0))
        set_value!(conf, key, value)
    end
    for (atom, alpha) in (("Ga", 9), ("As", 13))
        set_value!(conf, "orbitals", atom, "sp3dr2 sp3dr2 sp3dr2 sp3dr2")
        set_value!(conf, "n", atom, 3)
        set_value!(conf, "alpha", atom, alpha)
    end
    set_value!(conf, "nnaxes", "As", true)
    conf
end

function model_io_build(conf, comm)
    strc = Structure(conf)
    basis = Basis(strc, conf)
    ham = EffectiveHamiltonian([strc], [basis], comm, conf;
        rank=MPI.Comm_rank(comm), nranks=MPI.Comm_size(comm), write_summary=false)
    ham, [strc], [basis]
end

@testset "Model IO" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    dir = MPI.bcast(rank == 0 ? mktempdir() : nothing, 0, comm)
    try
        @test get_model_file(get_empty_config()) == "model.h5"
        @test !get_load_model(get_empty_config())
        @test !get_save_model(get_empty_config())
        for components in (:tb, :soc, :ewald, :ml, :ml_precalced, :combined)
            @testset "$components portable round trip" begin
                conf = model_io_config()
                if components in (:soc, :combined)
                    set_value!(conf, "soc", true)
                    set_value!(conf, "init_params", "SOC", "ones")
                end
                if components in (:ewald, :combined)
                    # Distorted conventional cell gives nonzero centered electrostatics.
                    poscar = joinpath(dir, "$components-POSCAR")
                    if rank == 0
                        write(poscar, """
                            GaAs distorted
                            5.65
                            1 0 0
                            0 1 0
                            0 0 1
                            Ga As
                            4 4
                            Direct
                            0 0 0
                            0 0.5 0.5
                            0.5 0 0.5
                            0.5 0.5 0
                            0.25 0.25 0.25
                            0.25 0.75 0.75
                            0.75 0.25 0.75
                            0.76 0.75 0.25
                            """)
                    end
                    MPI.Barrier(comm)
                    set_value!(conf, "poscar", poscar)
                    set_value!(conf, "method", "Ewald", "wolf")
                    set_value!(conf, "rcut", "Ewald", 8.0)
                    set_value!(conf, "centering", "Ewald", "structure")
                    set_value!(conf, "qeff", "Ga", 1.0)
                    set_value!(conf, "qeff", "As", -1.0)
                    set_value!(conf, "polarizability", "As", 0.2)
                    set_value!(conf, "charge_scale", "Ga", 0.7)
                    set_value!(conf, "charge_scale", "As", 0.9)
                end
                if components in (:ml, :ml_precalced, :combined)
                    set_value!(conf, "model", "ML", true)
                    set_value!(conf, "npoints", "ML", 4 * MPI.Comm_size(comm))
                    set_value!(conf, "ncluster", "ML", 1)
                    set_value!(conf, "sim_params", "ML", 0.4)
                    set_value!(conf, "env_scale", "ML", 1.3)
                    set_value!(conf, "strc_scale", "ML", 0.6)
                    set_value!(conf, "apply_distance_distortion", "ML", true)
                    set_value!(conf, "sim_mat", "ML", components == :ml_precalced)
                end
                # Export must remain usable after its source files disappear.
                radial_source = joinpath(dir, "$components-radial.dat")
                params_source = joinpath(dir, "$components-params.dat")
                if rank == 0
                    cp(conf("rllm_file"), radial_source)
                    cp(conf("init_params"), params_source)
                end
                MPI.Barrier(comm)
                set_value!(conf, "rllm_file", radial_source)
                set_value!(conf, "init_params", params_source)
                ham, strcs, bases = model_io_build(conf, comm)
                # Stand in for fitted weights, deliberately different from initialization.
                ham.models[1].params .*= 1.2
                for model in ham.models
                    if model isa Union{Hamster.HamiltonianKernel,Hamster.HamiltonianKernelPrecalced}
                        model.params .= 0.03
                    elseif model isa SOCModel
                        model.params .= 0.2
                    elseif model isa Hamster.EwaldOnsites
                        model.params[end] = 0.35
                        @test any(norm(h) > 1e-8 for h in Hamster.get_hr(model, ham.sp_mode, 1; apply_soc=ham.soc))
                    end
                end
                file = joinpath(dir, "$components.h5")
                save_model(ham, strcs, bases, conf, comm; filename=file, rank=rank)
                if rank == 0
                    rm(radial_source)
                    rm(params_source)
                end
                MPI.Barrier(comm)
                previous = rank == 0 ? read(file) : UInt8[]
                # A failed overwrite with missing source tables must preserve the export.
                @test_throws ErrorException save_model(ham, strcs, bases, conf, comm; filename=file, rank=rank)
                rank == 0 && @test read(file) == previous
                loaded = get_empty_config()
                set_value!(loaded, "poscar", conf("poscar"))
                set_value!(loaded, "verbosity", 0)
                # Deliberately conflicting settings must be replaced.
                set_value!(loaded, "alpha", "Ga", 2.0)
                load_model!(loaded, comm; filename=file, rank=rank)
                @test loaded("alpha", "Ga") == 9
                @test loaded("rcut") == 5
                @test get_model_file(loaded) == file
                @test loaded("poscar") == conf("poscar")
                restored, restored_strcs, restored_bases = model_io_build(loaded, comm)
                # Re-export also exercises radial input from an embedded HDF5 group.
                second_file = joinpath(dir, "$components-second.h5")
                save_model(restored, restored_strcs, restored_bases, loaded, comm; filename=second_file, rank=rank)
                @test Hamster.model_parameter_values(loaded, "tb", reverse(string.(ham.models[1].param_labels))) ≈ reverse(ham.models[1].params)
                @test_throws ErrorException Hamster.model_parameter_values(loaded, "tb", ["missing-parameter"])
                ks = zeros(3, 1)
                @test get_hamiltonian(restored, 1, ks) ≈ get_hamiltonian(ham, 1, ks)
                for (a,b) in zip(ham.models, restored.models)
                    @test a.params ≈ b.params
                end
                if components == :ewald
                    @test loaded("qeff", "Ga") == 1.0
                    @test loaded("centering", "Ewald") == "structure"
                    @test restored.models[2].pol_labels == ham.models[2].pol_labels
                end
                if components == :tb && MPI.Comm_size(comm) == 1
                    # Exercise a fresh process and working directory with only model.h5 + geometry.
                    script = joinpath(dir, "reload.jl")
                    write(script, """
                        using Hamster, MPI, HDF5
                        MPI.Init()
                        conf = get_empty_config()
                        set_value!(conf, "poscar", $(repr(conf("poscar"))))
                        set_value!(conf, "verbosity", 0)
                        load_model!(conf; filename=$(repr(file)))
                        strc = Structure(conf); basis = Basis(strc, conf)
                        ham = EffectiveHamiltonian([strc], [basis], MPI.COMM_WORLD, conf; write_summary=false)
                        h5open($(repr(joinpath(dir, "fresh.h5"))), "w") do f
                            f["H"] = get_hamiltonian(ham, 1, zeros(3,1))[1]
                        end
                        """)
                    run(Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$(pkgdir(Hamster)) $script`; dir=dir))
                    h5open(joinpath(dir, "fresh.h5"), "r") do f
                        @test read(f["H"]) ≈ get_hamiltonian(ham, 1, zeros(3,1))[1]
                    end
                end
                if rank == 0
                    h5open(file, "r") do f
                        @test read(f["metadata/timestamp"]) isa String
                        @test length(keys(f["model/rllm"])) > 0
                    end
                end
            end
        end
        if MPI.Comm_size(comm) == 1
            @testset "Optimization exports fitted weights" begin
                cd(dir) do
                    fixtures = abspath(joinpath(@__DIR__, "..", "optim", "test_files"))
                    conf = get_config(filename=joinpath(fixtures, "hconf"))
                    set_value!(conf, "poscar", joinpath(fixtures, "POSCAR_gaas"))
                    set_value!(conf, "rllm_file", joinpath(fixtures, "rllm.dat"))
                    set_value!(conf, "train_data", "Optimizer", joinpath(fixtures, "EIGENVAL_gaas"))
                    set_value!(conf, "val_data", "Optimizer", joinpath(fixtures, "EIGENVAL_gaas"))
                    set_value!(conf, "niter", "Optimizer", 2)
                    set_value!(conf, "verbosity", 0)
                    set_value!(conf, "save_model", true)
                    set_value!(conf, "model_file", "fitted.h5")
                    Hamster.main(comm, conf)
                    labels, params = Hamster.read_params("params.dat")[1:2]
                    stored = h5open("fitted.h5", "r") do f
                        Dict(zip(read(f["model/tb/labels"]), read(f["model/tb/params"])))
                    end
                    @test any(!isone, params)
                    @test [stored[string(label)] for label in labels] ≈ params
                end
            end
        end
        @testset "Load errors propagate to every rank" begin
            conf = get_empty_config()
            @test_throws ErrorException load_model!(conf, comm; filename=joinpath(dir,"missing.h5"), rank=rank)
            bad = joinpath(dir, "bad.h5")
            if rank == 0
                h5open(bad, "w") do f
                    f["version"] = "999"
                end
            end
            MPI.Barrier(comm)
            @test_throws ErrorException load_model!(conf, comm; filename=bad, rank=rank)
        end
    finally
        MPI.Barrier(comm)
        rank == 0 && rm(dir; recursive=true, force=true)
    end
end
