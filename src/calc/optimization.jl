"""
   run_calculation(::Val{:optimization}, conf::Config)

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
function run_calculation(::Val{:optimization}, comm, conf::Config; rank=0, nranks=1)
   train_config_inds, val_config_inds = get_config_index_sample(conf)
   
   if rank == 0
      write_to_file(train_config_inds, "train_config_inds")
      write_to_file(val_config_inds, "val_config_inds")
   end
   
   MPI.Bcast!(train_config_inds, comm, root=0)
   MPI.Bcast!(val_config_inds, comm, root=0)
   MPI.Barrier(comm)

   local_train_inds = split_indices_into_chunks(train_config_inds, nranks, rank=rank)
   local_val_inds = split_indices_into_chunks(val_config_inds, nranks, rank=rank)
   Rs = get_translation_vectors_for_hr_fit(conf)
   train_strcs = get_structures(conf, config_indices=local_train_inds, Rs=Rs, mode=get_train_mode(conf))
   train_bases = Basis[Basis(strc, conf) for strc in train_strcs]
   ham_train = EffectiveHamiltonian(train_strcs, train_bases, comm, conf, rank=rank, nranks=nranks)

   val_strcs = get_structures(conf, config_indices=local_val_inds, Rs=Rs, mode=get_val_mode(conf))
   val_bases = Basis[Basis(strc, conf) for strc in val_strcs]
   ham_val = EffectiveHamiltonian(val_strcs, val_bases, comm, conf, rank=rank, nranks=nranks, ml_data_points=get_ml_data_points(ham_train, conf))

   dl = DataLoader(local_train_inds, local_val_inds, length(train_bases[1]), length(train_bases[end]), conf)
   Nε, Nk = get_neig_and_nk(dl.train_data)
   optim = GDOptimizer(Nε, Nk, conf)
   prof = HamsterProfiler(3, conf)
   
   optimize_model!(ham_train, ham_val, optim, dl, prof, comm, conf, rank=rank, nranks=nranks)
   write_params(ham_train, conf)
   return prof
end