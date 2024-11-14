path = string(@__DIR__) * "/test_files/"

@testset "EffectiveHamiltonian" begin
    conf = get_empty_config()
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "save_rllm", false)
    eff_ham = EffectiveHamiltonian(conf)
    get_hr(eff_ham, 1)
    #@code_warntype get_hr(eff_ham, 1)
    rm("rllm.dat")
end