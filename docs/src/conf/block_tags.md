# [Config Blocks](@id block-labels)

Each block in the config file is structured as:

```bash
begin BLOCK_LABEL
    ...
end
```

Tags placed outside their designated blocks are ignored. The following blocks are available:

## Atom tags

```@autodocs
Modules = [Hamster]
Pages = ["atom_defaults.jl"]
```

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

## [Optimizer](@id optim-tags)

```@autodocs
Modules = [Hamster]
Pages = ["optim_defaults.jl"]
```

## [HyperOpt](@id hyperopt-tags)

```@autodocs
Modules = [Hamster]
Pages = ["hyperopt_defaults.jl"]
```