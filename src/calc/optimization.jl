"""
   run_calculation(::Val{:optimization}, comm, conf::Config; rank=0, nranks=1)

Runs the optimization process for an effective Hamiltonian model using the specified configuration.

# Arguments
- `conf::Config`: A configuration instance.

# Workflow
1. **Configuration Sampling**:
   - Splits the configuration indices into training (`train_config_inds`) and validation (`val_config_inds`) sets.
2. **Translation Vectors**:
   - Reads translation vectors `Rs` (if `hr_fit`).
3. **Training Data Preparation**:
   - Retrieves training structures (`train_strcs`) and constructs basis functions (`train_bases`).
   - Initializes the effective Hamiltonian (`ham_train`) for the training data.
4. **Validation Data Preparation**:
   - Retrieves validation structures (`val_strcs`) and constructs basis functions (`val_bases`).
   - Initializes the effective Hamiltonian (`ham_val`) for the validation data.
5. **Data Loader and Optimizer**:
   - Initializes a `DataLoader` (`dl`) with the training and validation configuration indices.
   - Sets up a gradient descent optimizer (`optim`) with the extracted parameters.
6. **Profiler**:
   - Creates a `HamsterProfiler` (`prof`) for profiling the optimization process.
7. **Model Optimization**:
   - Performs the optimization using `optimize_model!`, which iterates over the training and validation data to refine the model.
"""
function run_calculation(::Val{:optimization}, comm, conf::Config; rank=0, nranks=1, write_output=true)
   systems = get_systems(conf)
   
   train_config_inds, val_config_inds = get_config_inds_for_systems(systems, comm, conf, rank=rank, write_output=write_output)
   local_train_inds = split_indices_into_chunks(train_config_inds, nranks, rank=rank)
   local_val_inds = split_indices_into_chunks(val_config_inds, nranks, rank=rank)

   has_data = !isempty(local_train_inds) && (!isempty(local_val_inds) || !get_validate(conf))
   color = has_data ? 1 : nothing
   comm_active = MPI.Comm_split(comm, color, rank)

   all_has_data = MPI.gather(Int(has_data), comm, root=0)
   if rank == 0
      for (r, flag) in enumerate(all_has_data)
         if flag == 0
            append_output_line("Warning: Rank $(r-1) has no data and will be idle. Revise your parallelization settings!")
         end
      end
   end

   if has_data
      active_rank = MPI.Comm_rank(comm_active)
      active_size = MPI.Comm_size(comm_active)
      Rs = get_translation_vectors_for_hr_fit(conf)
      
      # EffectiveHamiltonian model for training set
      train_strcs = mapreduce(vcat, local_train_inds, init=Structure[]) do (system, train_inds)
         system_strcs = get_structures(conf, config_indices=train_inds, Rs=Rs, mode=get_train_mode(conf), system=system)
         if length(system_strcs) < length(train_inds)
            local_train_inds[system] = collect(1:length(system_strcs))
         end
         return system_strcs
      end
      train_bases = Basis[Basis(strc, conf, comm=comm_active) for strc in train_strcs]
      ham_train = EffectiveHamiltonian(train_strcs, train_bases, comm_active, conf, rank=active_rank, nranks=active_size)
      
      # EffectiveHamiltonian model for validation set
      val_strcs = mapreduce(vcat, local_val_inds, init=Structure[]) do (system, val_inds)
         system_strcs = get_structures(conf, config_indices=val_inds, Rs=Rs, mode=get_val_mode(conf), system=system)
         if length(system_strcs) < length(val_inds)
            local_val_inds[system] = collect(1:length(system_strcs))
         end
         return system_strcs
      end

      val_bases = Basis[Basis(strc, conf, comm=comm_active) for strc in val_strcs]
      ham_val = EffectiveHamiltonian(val_strcs, val_bases, comm_active, conf, rank=active_rank, nranks=active_size, ml_data_points=get_ml_data_points(ham_train, conf))
      Nε_train = get_number_of_bands_per_structure(train_bases, local_train_inds, soc=get_soc(conf))
      Nε_val = get_number_of_bands_per_structure(val_bases, local_val_inds, soc=get_soc(conf))

      combine_local_rllm_files(get_rllm_file(conf), comm_active; rank=active_rank, nranks=active_size)

      dl = DataLoader(local_train_inds, local_val_inds, Nε_train, Nε_val, conf)
      Nε, Nk = get_neig_and_nk(dl.train_data)
      optim = GDOptimizer(Nε, Nk, conf)
      prof = HamsterProfiler(3, conf)
      
      optimize_model!(ham_train, ham_val, optim, dl, prof, comm_active, conf, rank=active_rank, nranks=active_size)
   else
      prof = HamsterProfiler(3, conf)
   end
   MPI.Barrier(comm)
   if rank == 0 && write_output
      write_params(ham_train, conf)
   end
   return prof
end