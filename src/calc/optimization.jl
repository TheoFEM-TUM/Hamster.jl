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
   - On rank 0, optionally write optimized parameters to output files.

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

- `hamster.out` — standard Hamster output.  
- `hamster_out.h5` — HDF5 file containing:
  - Loss values over iterations
  - Timings
"""
function run_calculation(::Val{:optimization}, comm, conf::Config; rank=0, nranks=1, write_output=true)
   local_train_inds, local_val_inds = local_structure_indices(conf, comm, optimize=true, rank=rank, nranks=nranks,
                                                              write_output=write_output)
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
      
      optimize_model!(ham_train, ham_val, optim, dl, prof, comm_active, conf, rank=active_rank, nranks=active_size)
      if rank == 0 && write_output
         write_params(ham_train, conf)
      end
   else
      prof = HamsterProfiler(3, conf)
   end
   MPI.Barrier(comm)

   return prof
end