# Unit tests
split_line = Hamster.split_line

@testset "split_line" begin
    # Basic tests
    @test split_line("This is a test") == ["This", "is", "a", "test"]
    @test split_line("Hello World") == ["Hello", "World"]
    
    # Tests with multiple spaces
    @test split_line("This  is   a test") == ["This", "is", "a", "test"]
    @test split_line("  Julia   language  ") == ["Julia", "language"]
    
    # Edge cases
    @test split_line("") == []
    @test split_line(" ") == []
    @test split_line("   ") == []
    @test split_line("OneWord") == ["OneWord"]
    @test split_line("   OneWord   ") == ["OneWord"]
    
    # Tests with special characters
    @test split_line("Hello, world! How are you?") == ["Hello,", "world!", "How", "are", "you?"]
    @test split_line("Julia, Python, and C++") == ["Julia,", "Python,", "and", "C++"]

    # Test with `,`
    @test split_line("This,is,a,test", char=",") == ["This", "is", "a", "test"]
    # Test with custom delimiter with no matches
    @test split_line("No delimiters here", char=",") == ["No delimiters here"]

    # Test with custom delimiter at the start and end
    @test split_line(",Delimiters,at,start,and,end,", char=",") == ["Delimiters", "at", "start", "and", "end"]

    # Test with custom delimiter with consecutive delimiters
    @test split_line("Consecutive,,delimiters", char=",") == ["Consecutive", "delimiters"]

    # Test with mixed whitespace characters
    @test split_line("Mixed\twhitespace\ncharacters\r\n", char=r"\s") == ["Mixed", "whitespace", "characters"]

    # Test with different unicode characters as delimiters
    @test split_line("Unicode✨delimiters✨here", char="✨") == ["Unicode", "delimiters", "here"]
    @test split_line("Special*characters*in*delimiter", char="*") == ["Special", "characters", "in", "delimiter"]
    @test split_line("Escape\\nsequences\\nare\\ninteresting", char="\\n") == ["Escape", "sequences", "are", "interesting"]

    # Test with long string
    @test split_line("This is a very long string to test the splitting function with a default space delimiter") == ["This", "is", "a", "very", "long", "string", "to", "test", "the", "splitting", "function", "with", "a", "default", "space", "delimiter"]
end

@testset "Write and read to/from file" begin
    # Test 1: test random vector
    M = rand(8)
    write_to_file(M, "testfile")
    @test read_from_file("testfile.dat") == M


    # Test 2: test random matrix
    M = rand(8, 8)
    write_to_file(M, "testfile")
    @test read_from_file("testfile.dat") == M

    # Test 3: test random tensor
    M = rand(2, 2, 2)
    write_to_file(M, "testfile")
    @test read_from_file("testfile.dat") == M

    # Test 3: test another random tensor
    M = rand(2, 2, 2, 2)
    write_to_file(M, "testfile")
    @test read_from_file("testfile.dat") == M

    rm("testfile.dat")
end

@testset "Hamster logo" begin
    out = @capture_out Hamster.print_hamster()
    @test out == "\nWelcome to\n===========================================================================\n||   _   _                             _                         _   _   ||\n||  | | | |   __ _   _ __ ___    ___  | |_    ___   _ __        (_) | |  ||\n||  | |_| |  / _` | | '_ ` _ \\  / __| | __|  / _ \\ | '__|       | | | |  ||\n||  |  _  | | (_| | | | | | | | \\__ \\ | |_  |  __/ | |     _    | | | |  ||\n||  |_| |_|  \\__,_| |_| |_| |_| |___/  \\__|  \\___| |_|    (_)  _/ | |_|  ||\n||                                                            |__/       ||\n===========================================================================\nHamiltonian-learning\n    Approach for Multiscale Simulations \n        using a Transferable and Efficient Representation.\n\n\n"
end