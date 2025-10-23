@testset "Simple Universal Model" begin
    path = joinpath(@__DIR__, "test_files")
    strc_file = joinpath(path, "strc_file.h5") 
    eig_file = joinpath(path, "eig_file.h5")
    if isfile(strc_file); rm(strc_file); end
    if isfile(eig_file); rm(eig_file); end
    
    poscar_gaas = Hamster.read_poscar(joinpath(path, "POSCAR_gaas"))
    poscar_cspbrbr3 = Hamster.read_poscar(joinpath(path, "POSCAR_cspbbr3"))

    kp_gaas, Es_gaas, _ = Hamster.read_eigenval(joinpath(path, "EIGENVAL_gaas"))
    kp_cspbbr3, Es_cspbbr3, _ = Hamster.read_eigenval(joinpath(path, "EIGENVAL_cspbbr3"))

    # Generate dataset for structures
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

    # Generate dataset for eigenvalues
    h5open(eig_file, "cw") do f
        g1 = create_group(f, "cspbbr3")
        write(g1, "kpoints", kp_cspbbr3)
        write(g1, "eigenvalues", reshape(Es_cspbbr3[14:end, :], 17, 84, 1))

        g2 = create_group(f, "gaas")
        write(g2, "kpoints", kp_gaas)
        write(g2, "eigenvalues", reshape(Es_gaas, 48, 56, 1))
    end

    # Run calculation
    conf = get_config(filename = joinpath(path, "hconf_universal"))
    set_value!(conf, "rllm_file", joinpath(path, "rllm_universal.dat"))
    set_value!(conf, "train_data", "Optimizer", eig_file)
    set_value!(conf, "xdatcar", "Supercell", strc_file)
    #set_value!(conf, "val_data", "Optimizer", joinpath(path, "EIGENVAL_gaas"))

    prof = Hamster.main(comm, conf, rank=rank)
    @test mean(prof.L_train[:, end]) < 0.2
    rm("hamster.out"); rm("hamster_out.h5"); rm("params.dat"); rm(strc_file); rm(eig_file)
end