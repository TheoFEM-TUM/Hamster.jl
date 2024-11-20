"""
HamsterProfiler

A data structure for tracking and managing profiling information during an iterative process, including loss, timing, and verbosity settings.

# Fields
- `L_train::Matrix{Float64}`: A 2D array where each row corresponds to training loss values for a specific batch across iterations.
- `L_val::Vector{Float64}`: A 1D array containing validation loss values across iterations.
- `printeachbatch::Bool`: Indicates whether to print detailed status updates for each batch during training.
- `printeachiter::Int64`: Specifies the frequency (in number of iterations) at which status updates are printed.
- `timings::Array{Float64, 3}`: A 3D array storing timing information:
  - First dimension: batches.
  - Second dimension: iterations.
  - Third dimension: additional timing steps or phases within each iteration.
- `val_times::Vector{Float64}`: A 1D array containing validation timing information for each iteration.

# Usage
The `HamsterProfiler` struct is used in training workflows to:
- Record and track training and validation loss values.
- Measure and store timing data for profiling purposes.
- Control the verbosity and frequency of printed status updates.
"""
struct HamsterProfiler
    L_train :: Matrix{Float64}
    L_val :: Vector{Float64}
    printeachbatch :: Bool
    printeachiter :: Int64
    timings :: Array{Float64, 3}
    val_times :: Vector{Float64}
end

"""
    HamsterProfiler(Ntimes, conf=get_empty_config(); Nbatch=get_nbatch(conf), Niter=get_niter(conf), printeachbatch=get_printeachbatch(conf), printeachiter=get_printeachiter(conf))

A constructor function for initializing a `HamsterProfiler` instance.

# Arguments
- `Ntimes::Int`: The number of timing measurements to store for each batch and iteration.
- `conf::Dict`: A configuration dictionary (optional). Default is the result of `get_empty_config()`.
- `Nbatch::Int`: The number of batches (optional). Default is determined by the value of `get_nbatch(conf)`.
- `Niter::Int`: The number of iterations (optional). Default is determined by the value of `get_niter(conf)`.
- `printeachbatch::Bool`: A flag to determine whether to print detailed status for each batch (optional). Default is determined by the value of `get_printeachbatch(conf)`.
- `printeachiter::Int`: Specifies the frequency of printing status updates (optional). Default is determined by the value of `get_printeachiter(conf)`.

# Returns
- An instance of the `HamsterProfiler` struct.
"""
function HamsterProfiler(Ntimes, conf=get_empty_config(); Nbatch=get_nbatch(conf), Niter=get_niter(conf), printeachbatch=get_printeachbatch(conf), printeachiter=get_printeachiter(conf))
    HamsterProfiler(zeros(Nbatch, Niter), zeros(Niter), printeachbatch, printeachiter, zeros(Nbatch, Niter, Ntimes), zeros(Niter))
end

"""
print_train_status(prof, iter, batch_id; verbosity=1)

Prints the training status during an iterative training process, including loss, timing, and iteration progress.

# Arguments
- `prof`: A profiling object containing training data, configuration, and timing information. Expected fields include:
  - `L_train`: A 2D array where each row corresponds to batch loss values across iterations.
  - `timings`: A 3D array of timing data, where the first dimension corresponds to batch timing, the second to iteration, and the third to another dimension (e.g., steps or phases).
  - `printeachbatch`: A boolean indicating whether to print details for every batch.
  - `printeachiter`: A boolean indicating whether to print details for iterations only.
- `iter::Int`: The current iteration number.
- `batch_id::Int`: The current batch identifier (use `0` if not using batches).
- `verbosity::Int` (optional): Controls the level of detail in the output. Default is `1`.
"""
function print_train_status(prof, iter, batch_id; verbosity=1)
    Nbatch, Niter = size(prof.L_train)
    
    printit = decide_printit(batch_id, Nbatch, iter, prof.printeachbatch, prof.printeachiter; verbosity=verbosity)

    time_iter = sum(prof.timings[:, iter, :])
    time_left = sum(prof.timings[:, 1:iter, :]) / iter * (Niter - iter)
    if printit && prof.printeachbatch
        print_iteration_status(iter, Niter, batch_id, Nbatch, mean(prof.L_train[:, iter]), time_iter, time_left)
    elseif printit && !prof.printeachbatch
        print_iteration_status(iter, Niter, 0, 0, mean(prof.L_train[:, iter]), time_iter, time_left)
    end
end
"""
    print_val_start(; verbosity=1)

Prints a message indicating the start of the model validation process.

# Arguments
- `verbosity`: (optional) An integer controlling the level of output. If `verbosity > 0`, the message "Validating model..." will be printed. Default is `1`.
"""
print_val_start(; verbosity=1) = if verbosity > 0; println("Validating model..."); end

"""
    print_val_status(prof, iter; verbosity=1)

Prints the validation loss and the time taken for validation at a specific iteration during the training process.

# Arguments
- `prof`: A `HamsterProfiler` object that contains profiling information, including the validation loss (`L_val`) and validation times (`val_times`).
- `iter`: The current iteration number in the training process.
- `verbosity`: (optional) An integer controlling the level of output. If `verbosity > 0`, the validation status will be printed. Default is `1`.
"""
function print_val_status(prof, iter; verbosity=1)
    _, Niter = size(prof.L_train)
    if verbosity > 0
        println(@sprintf("Iteration: %d / %d | Val Loss: %.4f | Time: %.5f s", iter, Niter, prof.L_val[iter], prof.val_times[iter]))
    end
end

"""
print_iteration_status(iter, Niter, batch_id, Nbatch, L_train, time_iter, time_left)

Prints the current status of an iterative process, including iteration number, batch details, loss, and timing information.

# Arguments
- `iter::Int`: The current iteration number.
- `Niter::Int`: The total number of iterations.
- `batch_id::Int`: The current batch identifier (use `0` if not using batches).
- `Nbatch::Int`: The total number of batches (use `0` if not using batches).
- `L_train::Union{Float64, Int}`: The current training loss value. Use `0` if loss is not being reported.
- `time_iter::Float64`: Time elapsed for the current iteration, in seconds.
- `time_left::Float64`: Estimated time remaining to complete the process, in seconds.

# Behavior
The function adapts its printed output based on the values of `batch_id` and `L_train`:
1. **No batches and loss reported**:
   - Prints iteration number, total iterations, loss, iteration time, and estimated time left.
2. **No batches, no loss reported**:
   - Prints iteration number, total iterations, iteration time, and estimated time left.
3. **With batches and loss reported**:
   - Prints batch number, total batches, iteration number, total iterations, loss, iteration time, and estimated time left.
4. **With batches, no loss reported**:
   - Prints batch number, total batches, iteration number, total iterations, iteration time, and estimated time left.
"""
function print_iteration_status(iter, Niter, batch_id, Nbatch, L_train, time_iter, time_left)
    if L_train ≠ 0 && batch_id == 0
        println(@sprintf("Iteration: %d / %d | Loss: %.4f | Time: %.5f s | ETA: %.2f s", iter, Niter, L_train, time_iter, time_left))
    elseif batch_id == 0 && L_train == 0
        println(@sprintf("Iteration: %d / %d | Time: %.5f s | ETA: %.2f s", iter, Niter, time_iter, time_left))
    elseif L_train ≠ 0 && batch_id ≠ 0
        println(@sprintf("Batch %d / %d | Iteration: %d / %d | Loss: %.4f | Time: %.5f s | ETA: %.2f s", batch_id, Nbatch, iter, Niter, L_train, time_iter, time_left))
    elseif L_train == 0 && batch_id ≠ 0
        println(@sprintf("Batch %d / %d | Iteration: %d / %d | Time: %.5f s | ETA: %.2f s", batch_id, Nbatch, iter, Niter, time_iter, time_left))
    end
end

"""
print_start_message(prof)

Prints a message indicating the start of a run, along with the total number of iterations.

# Arguments
- `prof`: A HamsterProfiler instance.
""" 
function print_start_message(prof::HamsterProfiler; verbosity=1)
    Nbatch, Niter = size(prof.L_train)
    if verbosity > 0
        println("========================================")
        println("Run Starting!")
        println("Initializing HamsterProfiler...")
        println(@sprintf("Total Iterations: %d", Niter))
        println(@sprintf("Total Batches per Iteration: %d", Nbatch))
        println("========================================")
    end
end

"""
print_final_status(prof)

Prints the final loss, the total time elapsed, and a message indicating that the run has finished.

# Arguments
- `prof`: A HamsterProfiler instance used to compute `final_loss` and `total_time`.
"""
function print_final_status(prof; verbosity=1)
    total_time = sum(prof.timings)
    final_train_loss = mean(prof.L_train[:, end])
    final_val_loss = prof.L_val[end]
    if verbosity > 0
        println("")
        println("========================================")
        println("Run Finished!")
        println(@sprintf("Final Train Loss: %.6f", final_train_loss))
        if final_val_loss ≠ 0; println(@sprintf("Final Val Loss: %.6f", final_loss)); end
        println(@sprintf("Total Time: %.2f seconds", total_time))
        println("========================================")
    end
end

"""
decide_printit(batch_id, Nbatch, iter, printeachbatch, printeachiter; verbosity=verbosity)

Determines whether to print status updates based on the current iteration, batch, and verbosity settings.

# Arguments
- `batch_id::Int`: The current batch identifier.
- `Nbatch::Int`: The total number of batches.
- `iter::Int`: The current iteration number.
- `printeachbatch::Bool`: Whether to print at the end of each batch.
- `printeachiter::Int`: Interval for printing during iterations if `printeachbatch` is false.
- `verbosity::Int`: (Keyword argument) The verbosity level. Printing is disabled if `verbosity` is less than 1.

# Returns
- `Bool`: `true` if a status update should be printed, `false` otherwise.

# Behavior
1. If `verbosity < 1`, printing is disabled and the function returns `false`.
2. If `printeachbatch` is `true`, the function always returns `true`.
3. Otherwise, printing occurs only if:
   - The current batch is the last batch (`batch_id == Nbatch`), and
   - The current iteration number (`iter`) is a multiple of `printeachiter`.
"""
function decide_printit(batch_id, Nbatch, iter, printeachbatch, printeachiter; verbosity=verbosity)
    if verbosity < 1
        return false
    else
        if printeachbatch
            return true
        else
            return batch_id == Nbatch && mod(iter, printeachiter) == 0
        end
    end
end