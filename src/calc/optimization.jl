function get_subdirs(target_dir::String, systems::Vector{String})
   subdirs = [joinpath(target_dir, "DATA", d, "LOPTICS") for d in readdir(joinpath(target_dir, "DATA")) if isdir(joinpath(target_dir, "DATA", d))]
   matnames = [d for d in readdir(joinpath(target_dir, "DATA")) if isdir(joinpath(target_dir, "DATA", d))]
   exceptions =["plots","tmp"]
   subdirs = [subdirs[i] for i in 1:length(subdirs) if !(matnames[i] in exceptions)]
   matnames = [matnames[i] for i in 1:length(matnames) if !(matnames[i] in exceptions)]
   idx = Dict(b => i for (i, b) in enumerate(systems))
   perm = [idx[a] for a in matnames]
   subdirs = subdirs[perm]
   matnames = matnames[perm]
   return Dict(zip(matnames, subdirs))
end

function get_local_subdirs(subdirs::Vector{String}, matnames::Vector{String}, local_inds, config_inds) 
   local_length = length(local_inds)
   local_subdirs = [ "missing" for i in 1:local_length]
   if length(subdirs) < local_length
         error("The number of subdirectories is less than the configuration index. Please check your target directory and systems.")
   end
   for (system, ind) in config_inds
      if system in matnames
         sys_ind = findfirst(isequal(system), matnames)
         local_subdirs[ind] = subdirs[sys_ind]
      else
         error("System $system not found in the target directory subdirectories.")
      end

   end
   #sorted_subset = sort(filter(haskey(d), subdirs), by = d)
   return local_subdirs
end

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
   xdatcar_val = get_xdatcar_val(conf)
   #println("xdatcar_val: ", xdatcar_val)
   println("systems: ", systems)

   target_dir = get_target_directory(conf)
   
   train_config_inds, val_config_inds = get_config_inds_for_systems(systems, comm, conf, rank=rank, write_output=write_output)
   println("val_ratio: ", get_val_ratio(conf))
   if get_validate(conf) && get_val_ratio(conf) == 0
      systems_val = get_systems(conf, is_val=true)
      val_config_inds,_  = get_config_inds_for_systems(systems_val, comm, conf, rank=rank, write_output=write_output, is_val=true)
   end
   local_train_inds = split_indices_into_chunks(train_config_inds, nranks, rank=rank)
   local_val_inds = split_indices_into_chunks(val_config_inds, nranks, rank=rank)
   if target_dir != "missing"
      subdirs_dict = get_subdirs(target_dir, systems)
      local_subdirs = [subdirs_dict[k] for k in local_train_inds.keys if haskey(subdirs_dict, k)]
      #local_subdirs = split_indices_into_chunks(subdirs, nranks, rank=rank)
      #local_subdirs = collect(local_subdirs)
      #local_subdirs = reverse(local_subdirs) # Reverse the order of local_subdirs to match the order of systems
      #println("local systems subdirs for rank ", rank, ": ", systems[local_train_inds])
   else
      subdirs = ["missing" for i in 1:length(systems)]
      local_subdirs = split_indices_into_chunks(subdirs, nranks, rank=rank)
      local_subdirs = collect(local_subdirs)
   end
   has_data = !isempty(local_train_inds) && (!isempty(local_val_inds) || !get_validate(conf))
   if rank == 0
      println(!isempty(local_train_inds), " ", !isempty(local_val_inds), " ", get_validate(conf))
   end
   
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

      if target_dir != "missing"
         ham_train = EffectiveHamiltonian(train_strcs, train_bases, local_subdirs, comm_active, conf, rank=active_rank, nranks=active_size)
      else
         ham_train = EffectiveHamiltonian(train_strcs, train_bases, comm_active, conf, rank=active_rank, nranks=active_size)
      end
      # EffectiveHamiltonian model for validation set
      val_strcs = mapreduce(vcat, local_val_inds, init=Structure[]) do (system, val_inds)
         system_strcs = get_structures(conf, config_indices=val_inds, Rs=Rs, mode=get_val_mode(conf), system=system, is_val=true)
         if length(system_strcs) < length(val_inds)
            local_val_inds[system] = collect(1:length(system_strcs))
         end
         return system_strcs
      end

      val_bases = Basis[Basis(strc, conf, comm=comm_active) for strc in val_strcs]
      ham_val = EffectiveHamiltonian(val_strcs, val_bases, comm_active, conf, rank=active_rank, nranks=active_size, ml_data_points=get_ml_data_points(ham_train, conf), rllm_type="val")
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