# Config Blocks

Each block in the config file is structured as:

```bash
begin BLOCK_LABEL
    ...
end
```

Tags placed outside their designated blocks are ignored. The following blocks are available:

## Supercell

```@autodocs
Modules = [Hamster]
Pages = ["supercell_defaults.jl"]
```

## ML

```@autodocs
Modules = [Hamster]
Pages = ["ml_defaults.jl"]
```

## SOC

```@autodocs
Modules = [Hamster]
Pages = ["soc_defaults.jl"]
```

## Optim

```@autodocs
Modules = [Hamster]
Pages = ["optim_defaults.jl"]
```

## HyperOpt

```@autodocs
Modules = [Hamster]
Pages = ["hyperopt_defaults.jl"]
```