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

## Ewald onsite corrections

```@autodocs
Modules = [Hamster]
Pages = ["ewald_defaults.jl"]
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

## [DOS](@id dos-tags)

Settings of the [DOS, PDOS and COHP](@ref dos-post-processing) post-processing, which is switched on by the `dos` tag
in the `Options` block.

```@autodocs
Modules = [Hamster]
Pages = ["dos_defaults.jl"]
```

## [MD](@id md-tags)

Settings of the [molecular dynamics](@ref md-run), which is switched on by the `MD` block. The atom blocks need the
`valence` tag (electrons per atom in the TB bands).

```@autodocs
Modules = [Hamster]
Pages = ["md_defaults.jl"]
```
