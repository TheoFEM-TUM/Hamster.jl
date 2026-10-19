"""
    run_calculation(::Val{:optimization}, comm, conf::Config; rank=0, nranks=1)

Performs the optimization using gradient descent of an effective Hamiltonian model using the specified configuration.  
The routine iteratively adjusts model parameters to minimize the loss over a training set, while optionally validating performance on a separate validation set.

# Workflow

1. **Configuration Sampling**  
   - Partition configuration indices into training (`train_config_inds`) and validation (`val_config_inds`) sets.  
   - Distribute indices among parallel ranks.

2. **Translation Vectors (if `hr_fit`)**
   - Retrieve translation vectors `Rs` for high-rank fitting, used in constructing the Hamiltonian basis.

3. **Training Data Preparation**  
   - Load the training structures (`train_strcs`) from the selected configurations.  
   - Construct the corresponding basis functions (`train_bases`).  
   - Initialize an `EffectiveHamiltonian` (`ham_train`) with the training data.

4. **Validation Data Preparation**  
   - Load validation structures (`val_strcs`) and construct basis functions (`val_bases`).  
   - Initialize an `EffectiveHamiltonian` (`ham_val`) for validation, including only the relevant data points.

5. **Data Loader and Optimizer Initialization**  
   - Create a `DataLoader` (`dl`) containing training and validation datasets.  
   - Initialize a gradient descent optimizer (`optim`) using the extracted training parameters.

6. **Profiler Setup**  
   - Create a `HamsterProfiler` (`prof`) to record losses, timings, and parameter evolution during optimization.

7. **Model Optimization**  
   - Execute `optimize_model!`, iterating over training and validation data to update model parameters.  
   - On rank 0, write the parameters and the checkpoint every `write_every` iterations (see below).

# Required Inputs

- **Structural configurations**:  
  - `train_mode = md`: see `get_xdatcar` and `get_sc_poscar`  
  - `train_mode = pc`: see `get_poscar`
- **Eigenvalue training data**: see `get_train_data`
- **Eigenvalue validation data** (optional): see `get_val_data`  
  If not provided, the training data is split according to `val_ratio`.

# Input Tags (`Optimizer`)

- Configuration tags that specify how optimization is performed (see [here](@ref optim-tags)).

# Output Files

- `params.dat` (and `ml_params.dat` with an ML model) — the parameters with the lowest validation loss so far (without
  validation: the latest parameters). Written every `write_every` iterations (default: every iteration) and at the end.
  The files are replaced only when they are complete, so an interrupted run leaves valid parameter files.
- `checkpoint/` (tag `checkpoint`) — the latest parameters and `optimizer.h5` with the iteration, the Adam moments, the
  losses and the training and validation indices, written every `write_every` iterations and at the end.
- `hamster.out` — standard Hamster output.  
- `hamster_out.h5` — HDF5 file containing:
  - Loss values over iterations
  - Timings

# Continuing a Fit

With `restart = true` in the `Optimizer` block, the fit continues after the last iteration of the checkpoint, e.g.,
after the time limit of a job, with the same settings (`niter` can be increased). The models are initialized from the
checkpoint (including the ML data points), and the Adam moments, the learning rate schedule, the loss history and the
training and validation structures (one MD system) are restored, so the fit continues exactly as without
interruption. The models must be the same as in the interrupted run.
"""
function run_calculation(::Val{:optimization}, comm, conf::Config; rank=0, nranks=1, write_output=true)
   restart_dir = get_restart(conf) && write_output ? checkpoint_dir(conf) : ""
   isempty(restart_dir) || prepare_restart!(conf, restart_dir)
   local_train_inds, local_val_inds, train_inds, val_inds = local_structure_indices(conf, comm, optimize=true, rank=rank,
                                                                                    nranks=nranks, write_output=write_output)
   has_data = !isempty(local_train_inds) && (!isempty(local_val_inds) || !get_validate(conf))
   comm_active = active_communicator(has_data, comm, rank=rank)

   if has_data
      active_rank = MPI.Comm_rank(comm_active)
      active_size = MPI.Comm_size(comm_active)
      Rs = get_translation_vectors_for_hr_fit(conf)
      ham_train, _, train_bases = build_hamiltonian(conf, comm_active, local_train_inds, mode=get_train_mode(conf), Rs=Rs,
                                                    reindex=true, rank=active_rank, nranks=active_size, label="training set",
                                                    write_summary=write_output, training=true)
      ham_val, _, val_bases = build_hamiltonian(conf, comm_active, local_val_inds, mode=get_val_mode(conf), Rs=Rs, reindex=true,
                                                rank=active_rank, nranks=active_size, label="validation set",
                                                write_summary=write_output, ml_data_points=get_ml_data_points(ham_train, conf))
      Nε_train = get_number_of_bands_per_structure(train_bases, local_train_inds, soc=get_soc(conf))
      Nε_val = get_number_of_bands_per_structure(val_bases, local_val_inds, soc=get_soc(conf))

      combine_local_rllm_files(get_rllm_file(conf), comm_active; rank=active_rank, nranks=active_size)

      dl = DataLoader(local_train_inds, local_val_inds, Nε_train, Nε_val, conf)
      Nε, Nk = get_neig_and_nk(dl.train_data)
      optim = GDOptimizer(Nε, Nk, conf)
      prof = HamsterProfiler(3, conf)
      
      start_iter, best_L_val = 1, Inf
      if !isempty(restart_dir)
         last_iter, best_L_val = restore_optimizer!(optim, prof, ham_train, restart_dir, conf)
         start_iter = last_iter + 1
         if rank == 0 && get_verbosity(conf) > 0; println("Continuing the fit from $restart_dir after iteration $last_iter."); end
      end

      # params.dat: parameters with the lowest validation loss (the latest without validation); checkpoint: latest
      validate = get_validate(conf)
      write_every = get_write_every(conf)
      config_inds = length(train_inds) == 1 ? (first(values(train_inds)), first(values(val_inds))) : (Int64[], Int64[])
      best_written = Ref(false)
      after_iteration = function (iter, validated)
         (rank == 0 && write_output) || return nothing
         if validated && prof.L_val[iter] < best_L_val
            best_L_val = prof.L_val[iter]
            write_params_atomically(ham_train, conf)
            best_written[] = true
         elseif !validate && write_every > 0 && mod(iter, write_every) == 0
            write_params_atomically(ham_train, conf)
         end
         if (write_every > 0 && mod(iter, write_every) == 0) || iter == optim.Niter
            write_checkpoint(ham_train, optim, prof, iter, best_L_val, config_inds, conf)
         end
         return nothing
      end

      optimize_model!(ham_train, ham_val, optim, dl, prof, comm_active, conf, rank=active_rank, nranks=active_size,
                      start_iter=start_iter, after_iteration=after_iteration)
      if rank == 0 && write_output && !(validate && best_written[])
         write_params_atomically(ham_train, conf)
      end
   else
      prof = HamsterProfiler(3, conf)
   end
   MPI.Barrier(comm)

   return prof
end