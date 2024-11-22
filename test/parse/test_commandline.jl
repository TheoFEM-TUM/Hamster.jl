import Hamster: parse_commandline

@testset "CLI tests" begin
    # Test 1: Positional arguments only
    args1 = parse_commandline(["Hello", "World"])
    @test args1["pos_arg_1"] == "Hello" && args1["pos_arg_2"] == "World"

    # Test 2: Boolean flag argument
    args2 = parse_commandline(["build", "-v"])
    @test args2["pos_arg_1"] == "build" && args2["v"] == "true"

    # Test 3: Multiple keyword arguments
    args3 = parse_commandline(["--rcut", "8", "--conf", "hconf"])
    @test args3["rcut"] == "8" && args3["conf"] == "hconf"

    # Test 4: Combination 1
    args4 = parse_commandline(["-r", "Hello", "World", "--tag", "tag_value"])
    @test args4["r"] == "true" && args4["pos_arg_1"] == "Hello" && args4["pos_arg_2"] == "World" && args4["tag"] == "tag_value"

    # Test 5: Combination 2
    args5 = parse_commandline(["Hello", "-r", "--tag", "tag_value"])
    @test args5["r"] == "true" && args5["pos_arg_1"] == "Hello" && args5["tag"] == "tag_value"

    # Test 6: Combination 3
    args6 = parse_commandline(["--tag", "tag_value", "-r", "Hello", "World"])
    @test args6["r"] == "true" && args6["pos_arg_1"] == "Hello" && args6["pos_arg_2"] == "World" && args6["tag"] == "tag_value"

    # Test 7: Test concatenation if separated by ", "
    args7 = parse_commandline(["--tag", "val1,", "val2,", "val3"])
    @test args7["tag"] == "val1,val2,val3"

    # Test 8: Test a "--" argument without a value
    args8 = parse_commandline(["--help", "--tag", "val"])
    @test args8["help"] == "true" && args8["tag"] == "val"

    # Test 9: Test "--" argument at the end
    args9 = parse_commandline(["Hello", "World", "--help"])
    @test args9["help"] == "true" && args9["pos_arg_1"] == "Hello" && args9["pos_arg_2"] == "World"

    # Test 10: Test keyword argument with comma but no following argument
    args10 = parse_commandline(["--tag", "val,"])
    @test args10["tag"] == "val,"

    # Test 11: Test help flag in front of positional arguments
    args11 = parse_commandline(["--help", "Hello", "World"])
    @test args11["help"] == "true" && args11["pos_arg_1"] == "Hello" && args11["pos_arg_2"] == "World"

    # Test 12: Test h flag instead of help
    args12 = parse_commandline(["-h", "--tag", "val"])
    @test args12["help"] == "true" && args12["tag"] == "val"

    # Test 14: Test that a value with '-' in it is parsed correctly
    args14 = parse_commandline(["--tag", "Hello-World"])
    @test args14["tag"] == "Hello-World"
end