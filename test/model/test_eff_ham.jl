path = string(@__DIR__) * "/test_files/"

@testset "EffectiveHamiltonian" begin
    conf = get_empty_config()
    set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
    set_value!(conf, "save_rllm", false)
    eff_ham = EffectiveHamiltonian(conf)
    get_hr(eff_ham, 1)
    #@code_warntype get_hr(eff_ham, 1)
    @code_warnget_hr(h::AbstractMatrix, V::AbstractVector, mode=Val{:dense}; apply_soc=false)
    rm("rllm.dat")
end

conf = get_empty_config()
set_value!(conf, "poscar", joinpath(path, "POSCAR_gaas"))
set_value!(conf, "save_rllm", false)
eff_ham = EffectiveHamiltonian(conf)
get_hr(eff_ham, 1)
@code_warntype get_hr(eff_ham, 1)

@show typeof(eff_ham)
@code_warntype get_hr(eff_ham.models[1].hs[1], eff_ham.models[1].V, Val{:dense}, apply_soc=false)
rm("rllm.dat")

@show Hamster.get_sp_mode(conf) :: Type{Hamster.get_sp_mode(conf)} 