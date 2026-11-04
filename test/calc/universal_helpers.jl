"""
    write_universal_strc_file(strc_file)

Write the structures of CsPbBr3 and GaAs (one configuration each) as two systems to `strc_file`.
"""
function write_universal_strc_file(strc_file)
    path = joinpath(@__DIR__, "test_files")
    poscar_gaas = Hamster.read_poscar(joinpath(path, "POSCAR_gaas"))
    poscar_cspbrbr3 = Hamster.read_poscar(joinpath(path, "POSCAR_cspbbr3"))
    h5open(strc_file, "cw") do f
        g1 = create_group(f, "cspbbr3")
        write(g1, "positions", reshape(poscar_cspbrbr3.rs_atom, 3, 5, 1))
        write(g1, "lattice", reshape(poscar_cspbrbr3.lattice, 3, 3, 1))
        write(g1, "atom_types", poscar_cspbrbr3.atom_types)

        g2 = create_group(f, "gaas")
        write(g2, "positions", reshape(poscar_gaas.rs_atom, 3, 2, 1))
        write(g2, "lattice", reshape(poscar_gaas.lattice, 3, 3, 1))
        write(g2, "atom_types", poscar_gaas.atom_types)
    end
end

"""
    universal_standard_config(strc_file)

Config of a standard run with the systems CsPbBr3 and GaAs of `strc_file` (TB parameters start from one).
"""
function universal_standard_config(strc_file)
    path = joinpath(@__DIR__, "test_files")
    conf = get_config(filename = joinpath(path, "hconf_universal"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_universal.dat"))
    set_value!(conf, "xdatcar", "Supercell", strc_file)
    set_value!(conf, "init_params", "ones")
    set_value!(conf, "kpoints", "gamma")
    set_value!(conf, "write_hr", true)
    return conf
end
