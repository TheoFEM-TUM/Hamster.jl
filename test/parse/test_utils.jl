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

@testset "get_rank_filename tests" begin
    @test Hamster.get_rank_filename("data.h5", 3) == "tmp/data_3.h5"

    @test Hamster.get_rank_filename("output.txt", 1) == "tmp/output_1.txt"

    @test Hamster.get_rank_filename("/home/user/sim/results.hdf5", 4) == "tmp/results_4.hdf5"

    @test Hamster.get_rank_filename("matrixfile", 10) == "tmp/matrixfile_10"

    @test Hamster.get_rank_filename("data/run1/sample.h5", 7) == "tmp/sample_7.h5"

    @test Hamster.get_rank_filename("data.h5", 2; temp=false) == "data_2.h5"

    @test Hamster.get_rank_filename("rawdata", 42; temp=false) == "rawdata_42"

    @test Hamster.get_rank_filename("archive.tar.gz", 8) == "tmp/archive.tar_8.gz"

    @test Hamster.get_rank_filename("foo.h5", 0) isa String
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

@testset "Eigen output file" begin
    mktempdir() do dir
        file = joinpath(dir, "eval_eigenval.h5")

        # Test 1: Named systems, configurations are appended in the order they are written
        Es_A7, Es_A2, Es_B1 = rand(2, 3), rand(2, 3), rand(4, 3)
        vs_A7, vs_A2, vs_B1 = rand(ComplexF64, 2, 2, 3), rand(ComplexF64, 2, 2, 3), rand(ComplexF64, 4, 4, 3)
        Hamster.append_eigen_output(Es_A7, vs_A7, "Cs_A", 7, filename=file, save_vecs=true)
        Hamster.append_eigen_output(Es_B1, vs_B1, "B", 1, filename=file, save_vecs=true)
        Hamster.append_eigen_output(Es_A2, vs_A2, "Cs_A", 2, filename=file, save_vecs=true)
        h5open(file, "r") do f
            @test sort(keys(f)) == ["B", "Cs_A"]
            @test read(f["Cs_A/config_inds"]) == [7, 2]
            @test read(f["Cs_A/eigenvalues"]) == cat(Es_A7, Es_A2, dims=3)
            @test read(f["Cs_A/eigenvectors"]) == cat(vs_A7, vs_A2, dims=4)
            @test read(f["B/config_inds"]) == [1]
            @test read(f["B/eigenvalues"]) == reshape(Es_B1, 4, 3, 1)
            @test read(f["B/eigenvectors"]) == reshape(vs_B1, 4, 4, 3, 1)
        end
        rm(file)

        # Test 2: Unnamed system is written to the file root, eigenvectors only if save_vecs
        Es_3, Es_5 = rand(2, 3), rand(2, 3)
        Hamster.append_eigen_output(Es_3, zeros(ComplexF64, 1, 1, 1), "", 3, filename=file)
        Hamster.append_eigen_output(Es_5, zeros(ComplexF64, 1, 1, 1), "", 5, filename=file)
        h5open(file, "r") do f
            @test sort(keys(f)) == ["config_inds", "eigenvalues"]
            @test read(f["config_inds"]) == [3, 5]
            @test read(f["eigenvalues"]) == cat(Es_3, Es_5, dims=3)
        end
    end
end

@testset "parse_lines_as_array tests" begin
    # Test 1: Simple float parsing
    lines = [["1.0", "2.0", "3.0"], ["4.0", "5.0", "6.0"]]
    result = Hamster.parse_lines_as_array(lines)
    @test result == [1.0 2.0 3.0; 4.0 5.0 6.0]

    # Test 2: Subrange with i1 and i2
    result = Hamster.parse_lines_as_array(lines, i1=2, i2=3)
    @test result == [2.0 3.0; 5.0 6.0]

    # Test 3: Parsing integers
    lines_int = [["1", "2", "3"], ["4", "5", "6"]]
    result = Hamster.parse_lines_as_array(lines_int, type=Int)
    @test result == [1 2 3; 4 5 6]
end