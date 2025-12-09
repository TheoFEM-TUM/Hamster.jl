# Basis

```@autodocs
Modules = [Hamster]
Pages = ["adaptive_intp.jl", "basis.jl", "index.jl", "label.jl", "orbconfig.jl", "orbital.jl", "overlap.jl", "param.jl", "rllm.jl", "sh_transforms.jl", "sper_harm.jl"]
```

# Structure

```@autodocs
Modules = [Hamster]
Pages = ["grid.jl", "ion.jl", "lattice.jl", "methods.jl", "sk_transform.jl", "structure.jl", "supercell.jl", "vec.jl"]
Filter = x -> !(x in [Hamster.run_calculation])
```

# TB

```@autodocs
Modules = [Hamster]
Pages = ["eff_ham.jl", "ham.jl", "ham_grad.jl", "ham_write.jl", "model/model.jl"]
```

# SOC

```@autodocs
Modules = [Hamster]
Pages = ["soc_matrix.jl", "soc_model.jl", "soc_utils.jl"]
```

# ML

```@autodocs
Modules = [Hamster]
Pages = ["descriptor.jl", "kernel.jl"]
```

# Optimization

```@autodocs
Modules = [Hamster]
Pages = ["adam.jl", "data.jl", "gd_optimizer.jl", "loss.jl", "optimize.jl", "profiler.jl"]
```

# Parsing functions

```@autodocs
Modules = [Hamster]
Pages = ["commandline.jl", "eigenval.jl", "poscar.jl", "parse/utils.jl", "wannier90.jl", "xdatcar.jl"]
```

# Main

```@autodocs
Modules = [Hamster]
Pages = ["main.jl"]
```

# Config

```@autodocs
Modules = [Hamster]
Pages = ["conf/read_config.jl", "conf/config.jl"]
```

# Output

```@autodocs
Modules = [Hamster]
Pages = ["output.jl"]
```