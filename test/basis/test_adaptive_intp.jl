using CubicSplines

const int_tol = 1e-3

@testset "Adaptive interpolation" begin
    # Test 1: test function 1
    conf = get_empty_config()
    set_value!(conf, "itp_xmax", 15)
    set_value!(conf, "itp_Ninit", 10)
    set_value!(conf, "itp_Nmax", 300)
    set_value!(conf, "itp_tol", 1e-6)
    f1(x) = (10*x - 7*x^2 + 2*x^3 - 3*x^4)*exp(-1.5*x)
    intp = Hamster.AdaptiveInterpolator(conf)

    @test intp.xmax == 15
    @test intp.Ninit == 10
    @test intp.tol == 1e-6
    @test intp.Nmax == 300
    xs, ys = Hamster.interpolate_f(intp, f1)
    f1_int = CubicSpline(xs, ys)

    x_trial = 15 .* rand(100)
    @test mean(abs.(f1_int.(x_trial) .- f1.(x_trial))) < int_tol

    # Test 2: test function 2
    f2(x) = (3x^4 - 2x^3 + 5x^2 - 7x + 1) * exp(-x^2)

    xs2, ys2 = Hamster.interpolate_f(intp, f2)
    f2_int = CubicSpline(xs2, ys2)

    x_trial = 15 .* rand(100)
    @test mean(abs.(f2_int.(x_trial) .- f2.(x_trial))) < int_tol

    # Test 3: test function 3
    f3(x) = (-2.5*x^5 + 3x^4 - 2x^3 + 5x^2 - 7x) * exp(-0.4*x^2)
    
    xs3, ys3 = Hamster.interpolate_f(intp, f3)
    f3_int = CubicSpline(xs3, ys3)

    x_trial = 15 .* rand(100)
    @test mean(abs.(f3_int.(x_trial) .- f3.(x_trial))) < int_tol
end