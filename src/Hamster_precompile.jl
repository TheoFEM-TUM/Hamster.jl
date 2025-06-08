using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    v = rand(3); w = rand(3); dv = rand(3); dw = rand(3); t = rand(3)

    Rs = rand(1:3, 3, 5)
    Rs_float = rand(3, 5)
    ks = rand(3, 5)
    Hrs = Matrix{Float64}[rand(4, 4) for R in 1:5]
    Hr_sp = [sprand(4, 4, 0.1) for R in 1:5]
    Hks = [Hermitian(rand(ComplexF64, 4, 4)) for i in 1:5]
    @compile_workload begin
        Hamster.normdiff(v, w)
        Hamster.normdiff(v, w, t)
        Hamster.normdiff(v, w, dv, dw, t)

        get_hamiltonian(Hrs, Rs, ks, Val{:dense})
        get_hamiltonian(Hrs, Rs_float, ks, Val{:dense})
        get_hamiltonian(Hr_sp, Rs, ks, Val{:dense})
        get_hamiltonian(Hr_sp, Rs_float, ks, Val{:dense})
        get_hamiltonian(Hrs, Rs, ks)
        get_hamiltonian(Hrs, Rs_float, ks)
        get_hamiltonian(Hr_sp, Rs, ks)
        get_hamiltonian(Hr_sp, Rs_float, ks)
        get_hamiltonian(Hr_sp, Rs, ks, Val{:sparse})
        get_hamiltonian(Hr_sp, Rs_float, ks, Val{:sparse})
        diagonalize(Hks)
    end
end