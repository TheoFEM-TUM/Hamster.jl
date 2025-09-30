@testset "Profiler validation prints" begin
    prof = HamsterProfiler(3)
    # Test 1: verbosity > 0
    val_print = @capture_out Hamster.print_val_start(prof, 1, verbosity=1)
    @test val_print == "   Validating model...\n"
    
    #Test 2: verbosity = 0
    val_print = @capture_out Hamster.print_val_start(prof, 1, verbosity=0)
    @test val_print == ""

    # Test 3: verbosity > 0
    val_print = @capture_out Hamster.print_val_status(prof, 1)
    @test val_print == "   Iteration: 1 / 1 | Val Loss: 0.0000 | Time: 0.00000 s\n"

    # Test 4: verbosity = 0
    val_print = @capture_out Hamster.print_val_status(prof, 1, verbosity=0)
    @test val_print == ""
end

@testset "Profiler training prints" begin
    prof = HamsterProfiler(3)

    # Test 1: test init print
    init_print = @capture_out Hamster.print_start_message(prof; verbosity=1)
    @test init_print == "========================================\nRun Starting!\nInitializing HamsterProfiler...\nTotal Iterations: 1\nTotal Batches per Iteration: 1\n========================================\n"
    init_print = @capture_out Hamster.print_start_message(prof; verbosity=0)
    @test init_print == ""

    prof.L_train[1, 1] = 0.3
    prof.timings[1, 1, :] .= [1, 2, 3]

    # Test 2: test final print
    final_print = @capture_out Hamster.print_final_status(prof, verbosity=1)
    @test final_print == "\n========================================\nRun Finished!\nFinal Train Loss: 0.300000\nTotal Time: 6.00 seconds\n========================================\n"
    final_print = @capture_out Hamster.print_final_status(prof, verbosity=0)
    @test final_print == ""

    # Test 3: test intermediate print, with and without batch
    status_print = @capture_out Hamster.print_train_status(prof, 1, 1)
    @test status_print == "Iteration: 1 / 1 | Loss: 0.3000 | Time: 6.00000 s | ETA: 0.00 s\n"

    prof = HamsterProfiler(3, printeachbatch=true)
    prof.L_train[1, 1] = 0.4
    prof.timings[1, 1, :] .= [2, 3, 4]
    status_print = @capture_out Hamster.print_train_status(prof, 1, 1)
    @test status_print == "Batch 1 / 1 | Iteration: 1 / 1 | Loss: 0.4000 | Time: 9.00000 s | ETA: 0.00 s\n"

    # Test 4: test status print without loss
    status_print = @capture_out Hamster.print_iteration_status(1, 10, 0, 0, 0, 0.5, 1)
    @test status_print == "Iteration: 1 / 10 | Time: 0.50000 s | ETA: 1.00 s\n"
    status_print = @capture_out Hamster.print_iteration_status(1, 10, 1, 5, 0, 0.5, 1)
    @test status_print == "Batch 1 / 5 | Iteration: 1 / 10 | Time: 0.50000 s | ETA: 1.00 s\n"
end

@testset "Profiler decide_printit" begin
    # Test 1: true for every iteration with interval 1; always false for verbosity=0
    @test all([Hamster.decide_printit(1, 1, iter, false, 1) for iter in 1:5])
    @test all([Hamster.decide_printit(1, 1, iter, false, 1, verbosity=0) for iter in 1:5]) == false

    # Test 2: true for every second iteration with interval 2
    @test [Hamster.decide_printit(1, 1, iter, false, 2) for iter in 1:4] == [false, true, false, true]

    # Test 3: true for every batch and ignores iter interval
    @test all([Hamster.decide_printit(batch_id, 1, iter, true, 1) for iter in 1:3 for batch_id in 1:4])
    @test all([Hamster.decide_printit(batch_id, 1, iter, true, 2) for iter in 1:3 for batch_id in 1:4])
end

@testset "Profiler save to file" begin
    filename = "test_prof.h5"

    # Step 1: Write some unrelated dataset
    h5open(filename, "w") do file
        file["unrelated_data"] = [1, 2, 3, 4, 5]
    end

    # Step 2: Save profiler (should not remove unrelated_data)
    prof = HamsterProfiler(
        rand(3, 3),
        rand(3),
        true,
        42,
        rand(2, 2, 2),
        rand(4),
        rand(5, 5)
    )
    Hamster.save(prof, filename=filename)

    # Step 3: Check both profiler fields AND unrelated dataset still exist
    h5open(filename, "r") do file
        @test haskey(file, "unrelated_data")
        @test read(file["unrelated_data"]) == collect(1:5)

        @test read(file["L_train"]) == prof.L_train
        @test read(file["L_val"]) == prof.L_val
        @test read(file["timings"]) == prof.timings
        @test read(file["val_times"]) == prof.val_times
        @test read(file["param_values"]) == prof.param_values

        attrs = attributes(file)
        @test read(attrs["printeachbatch"]) == prof.printeachbatch
        @test read(attrs["printeachiter"]) == prof.printeachiter
    end

    rm(filename; force=true)

    Hamster.save(prof, 1, filename=filename)
    @test !isfile(filename)
end
