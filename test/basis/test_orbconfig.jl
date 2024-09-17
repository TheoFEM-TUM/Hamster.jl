@testset "Orbital configuration" begin
    # Test 1: Test conjugates
    sym = Hamster.SymOrb()
    def = Hamster.DefOrb()
    mirr = Hamster.MirrOrb()


    @test typeof(Hamster.conjugate(sym)) <: Hamster.SymOrb
    @test typeof(Hamster.conjugate(def)) <: Hamster.MirrOrb
    @test typeof(Hamster.conjugate(mirr)) <: Hamster.DefOrb
end