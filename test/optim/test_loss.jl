@testset "Normal loss" begin
    # Test 1: MSE
    y = rand(2, 4)
    ŷ = rand(2, 4)

    mse(y, ŷ) = mean(@. (y - ŷ)^2)
    mae(y, ŷ) = mean(@. abs(y - ŷ))

    loss_1 = Loss(2)
    @test loss_1(y, ŷ) ≈ mse(y, ŷ)
    @test Hamster.backward(loss_1, y, ŷ) ≈ FiniteDiff.finite_difference_gradient(y->loss_1(y, ŷ), y)

    # Test 2: MAE
    y = rand(2, 4)
    ŷ = rand(2, 4)

    loss_2 = Loss(1)
    @test loss_2(y, ŷ) ≈ mae(y, ŷ)
    @test Hamster.backward(loss_2, y, ŷ) ≈ FiniteDiff.finite_difference_gradient(y->loss_2(y, ŷ), y)

    # Test 3: wMSE
    y = rand(2, 4)
    ŷ = rand(2, 4)
    wE = rand(2)
    wk = rand(4)

    loss_3 = Loss(wE, wk, 2)
    @test loss_3(y, ŷ) ≈ 1/(sum(wk) * sum(wE))*( (y .- ŷ).^2 * wk ) ⋅ wE
    @test Hamster.backward(loss_3, y, ŷ) ≈ FiniteDiff.finite_difference_gradient(y->loss_3(y, ŷ), y)

    # Test 4: wMAE
    y = rand(2, 4)
    ŷ = rand(2, 4)
    wE = rand(2)
    wk = rand(4)

    loss_4 = Loss(wE, wk, 1)
    @test loss_4(y, ŷ) ≈ 1/(sum(wk) * sum(wE))*( abs.(y .- ŷ) * wk ) ⋅ wE
    @test Hamster.backward(loss_4, y, ŷ) ≈ FiniteDiff.finite_difference_gradient(y->loss_4(y, ŷ), y)

    # Test 5: MAE from config
    conf = get_empty_config()
    set_value!(conf, "loss", "Optimizer", "MAE")
    loss_5 = Loss(2, 4, conf)
    @test loss_5.n == 1
    @test loss_5.N == 8
    @test loss_5.wE == [1, 1]
    @test loss_5.wk == [1, 1, 1, 1]

    # Test 6: MSE from config
    conf = get_empty_config()
    set_value!(conf, "loss", "Optimizer", "MSE")
    loss_6 = Loss(2, 4, conf)
    @test loss_6.n == 2
    @test loss_6.N == 8
    @test loss_6.wE == [1, 1]
    @test loss_6.wk == [1, 1, 1, 1]

    # Test 7: weights as array
    conf = get_empty_config()
    set_value!(conf, "wE", "Optimizer", "1 2")
    set_value!(conf, "wk", "Optimizer", "3 4 5 6")

    loss_7 = Loss(2, 4, conf)
    @test loss_7.wE == [1, 2]
    @test loss_7.wk == [3, 4, 5, 6]

    # Test 8: individual weights
    conf = get_empty_config()
    set_value!(conf, "wE_2", "Optimizer", 2)
    set_value!(conf, "wk_2", "Optimizer", 2)
    set_value!(conf, "wk_4", "Optimizer", 4)

    loss_8 = Loss(2, 4, conf)
    @test loss_8.wE == [1, 2]
    @test loss_8.wk == [1, 2, 1, 4]

    # Test 9: combine and overwrite
    conf = get_empty_config()
    set_value!(conf, "wE", "Optimizer", "2 1")
    set_value!(conf, "wE_2", "Optimizer", 2)
    set_value!(conf, "wk", "Optimizer", "2 2 2 2")
    set_value!(conf, "wk_2", "Optimizer", 3)
    set_value!(conf, "wk_4", "Optimizer", 3)

    loss_9 = Loss(2, 4, conf)
    @test loss_9.wE == [2, 2]
    @test loss_9.wk == [2, 3, 2, 3]

    # Test 10: test Regularization without barrier
    x = rand(10)
    reg_1 = Regularization(0., 1e-5, 1)

    @test reg_1(x) ≈ 1e-5 * sum(x)
    @test Hamster.backward(reg_1, x) ≈ FiniteDiff.finite_difference_gradient(x->reg_1(x), x)

    reg_2 = Regularization(0., 1e-6, 2)
    @test reg_2(x) ≈ 1e-6 * sum(x.^2)
    @test Hamster.backward(reg_2, x) ≈ FiniteDiff.finite_difference_gradient(x->reg_2(x), x)

    # Test 11: Test Regularization with barrier
    x = rand(10)
    reg_3 = Regularization(1.1, 1e-5, 2)
    @test reg_3(x) ≈ 0
    @test Hamster.backward(reg_3, x) ≈ FiniteDiff.finite_difference_gradient(x->reg_3(x), x)

    x_2 = rand(10) .+ 1.2
    @test reg_3(x_2) ≈ 1e-5 * sum((x_2 .- 1.1).^2)
    @test Hamster.backward(reg_3, x_2) ≈ FiniteDiff.finite_difference_gradient(x->reg_3(x), x_2)

    x_3 = rand(10)
    reg_4 = Regularization(0.5, 1e-5, 2)
    @test reg_4(x_3) ≈ 1e-5 * mapreduce(y -> abs(y) > 0.5 ? (y-0.5)^2 : 0., +, x_3)
    @test Hamster.backward(reg_4, x_3) ≈ FiniteDiff.finite_difference_gradient(x->reg_4(x), x_3)

    # Test 12: Test Regularization from conf
    conf = get_empty_config()
    set_value!(conf, "lambda", "Optimizer", 1e-5)
    set_value!(conf, "barrier", "Optimizer", 1)
    set_value!(conf, "lreg", "Optimizer", 1)
    reg = Regularization(conf)
    @test reg.λ == 1e-5
    @test reg.b == 1
    @test reg.n == 1

    # Test 13: Test weight ranges
    @test Hamster.get_weight_index_from_key("wk_1-3") == 1:3
    @test Hamster.get_weight_index_from_key("wE_5-9") == 5:9
    @test Hamster.get_weight_index_from_key("wE_5") == 5:5

    conf = get_empty_config()
    set_value!(conf, "wE_5-8", "Optimizer", 0.1)
    set_value!(conf, "wE_5", "Optimizer", 0.2)
    set_value!(conf, "wk", "Optimizer", "2 2 2 2")
    set_value!(conf, "wk_3-4", "Optimizer", 1)

    loss_13 = Loss(8, 4, conf)
    @test loss_13.wE == [1, 1, 1, 1, 0.2, 0.1, 0.1, 0.1]
    @test loss_13.wk == [2, 2, 1, 1]
end

@testset "Hr loss" begin
    mse(y, ŷ) = mean(@. (y - ŷ)^2)
    mae(y, ŷ) = mean(@. abs(y - ŷ))
    Hr_mse(Hr_1, Hr_2) = mean([mse(H1, H2) for (H1, H2) in zip(Hr_1, Hr_2)])
    Hr_mae(Hr_1, Hr_2) = mean([mae(H1, H2) for (H1, H2) in zip(Hr_1, Hr_2)])

    Hr_1 = [rand(4, 4) for _ in 1:4]
    Hr_2 = [rand(4, 4) for _ in 1:4]
    
    # Test 1: test MAE loss
    loss = Loss(1)
    @test Hr_mae(Hr_1, Hr_2) ≈ loss(Hr_1, Hr_2)
    dL_dHr = Hamster.backward(loss, Hr_1, Hr_2)
    dL_dHr_true = [FiniteDiff.finite_difference_gradient(Hr->mae(Hr, Hr_2[i]), Hr_1[i])/length(Hr_1) for i in eachindex(Hr_1)]
    @test all([dL_dHr ≈ dL_dHr_true for i in eachindex(dL_dHr)])

    # Test 2: test MSE loss
    loss = Loss(2)
    @test Hr_mse(Hr_1, Hr_2) ≈ loss(Hr_1, Hr_2)
    dL_dHr = Hamster.backward(loss, Hr_1, Hr_2)
    dL_dHr_true = [FiniteDiff.finite_difference_gradient(Hr->mse(Hr, Hr_2[i]), Hr_1[i])/length(Hr_1) for i in eachindex(Hr_1)]
    @test all([dL_dHr ≈ dL_dHr_true for i in eachindex(dL_dHr)])
end