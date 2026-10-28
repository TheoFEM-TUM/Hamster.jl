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

@testset "Collapse output files" begin
    # Test 1: Several systems, files are stacked per system and sorted by config index
    mktempdir() do dir
        cd(dir) do
            mkdir("tmp")
            Es_A1, Es_A3, Es_B2 = rand(2, 3), rand(2, 3), rand(4, 3)
            vs_A1 = rand(ComplexF64, 2, 2, 3)
            write_to_file(Es_A3, "tmp/Es_Cs_A_3"); write_to_file(Es_A1, "tmp/Es_Cs_A_1"); write_to_file(Es_B2, "tmp/Es_B_2")
            write_to_file(vs_A1, "tmp/vs_Cs_A_1")
            Hamster.collapse_files_with("Es")
            Es_A = read_from_file("Es_Cs_A.dat")
            @test size(Es_A) == (2, 3, 2)
            @test Es_A[:, :, 1] == Es_A1
            @test Es_A[:, :, 2] == Es_A3
            @test read_from_file("Es_B.dat")[:, :, 1] == Es_B2
            @test h5read("eval_config_inds.h5", "Cs_A") == [1, 3]
            @test h5read("eval_config_inds.h5", "B") == [2]
            @test !isfile("Es_Cs_A_1.dat") && !isfile("vs_Cs_A.dat")
        end
    end

    # Test 2: Single system with empty name, higher-dimensional data
    mktempdir() do dir
        cd(dir) do
            mkdir("tmp")
            Es_5, Es_2 = rand(2, 2, 2), rand(2, 2, 2)
            write_to_file(Es_5, "tmp/Es__5"); write_to_file(Es_2, "tmp/Es__2")
            Hamster.collapse_files_with("Es")
            Es = read_from_file("Es.dat")
            @test Es[:, :, :, 1] == Es_2
            @test Es[:, :, :, 2] == Es_5
            @test h5read("eval_config_inds.h5", "config_inds") == [2, 5]
        end
    end

    # Test 3: collect_output_files copies a single file and collapses several files
    mktempdir() do dir
        cd(dir) do
            mkdir("tmp")
            Es_1, vs_1 = rand(2, 3), rand(ComplexF64, 2, 2, 3)
            write_to_file(Es_1, "tmp/Es__1"); write_to_file(vs_1, "tmp/vs__1")
            Hamster.collect_output_files(["Es", "vs"])
            @test read_from_file("Es.dat") == Es_1
            @test read_from_file("vs.dat") == vs_1
            @test !isfile("eval_config_inds.h5")

            rm("Es.dat"); rm("vs.dat"); rm("tmp", recursive=true); mkdir("tmp")
            Es_2, Es_4 = rand(2, 3), rand(2, 3)
            write_to_file(Es_4, "tmp/Es__4"); write_to_file(Es_2, "tmp/Es__2")
            Hamster.collect_output_files(["Es", "vs"])
            Es = read_from_file("Es.dat")
            @test Es[:, :, 1] == Es_2 && Es[:, :, 2] == Es_4
            @test h5read("eval_config_inds.h5", "config_inds") == [2, 4]
            @test !isfile("eval_eigenval.h5")
        end
    end

    # Test 4: collect_output_files writes named systems into one h5 file
    mktempdir() do dir
        cd(dir) do
            mkdir("tmp")
            Es_A7, Es_A2, Es_B1 = rand(2, 3), rand(2, 3), rand(4, 3)
            vs_A7, vs_A2, vs_B1 = rand(ComplexF64, 2, 2, 3), rand(ComplexF64, 2, 2, 3), rand(ComplexF64, 4, 4, 3)
            write_to_file(Es_A7, "tmp/Es_Cs_A_7"); write_to_file(Es_A2, "tmp/Es_Cs_A_2"); write_to_file(Es_B1, "tmp/Es_B_1")
            write_to_file(vs_A7, "tmp/vs_Cs_A_7"); write_to_file(vs_A2, "tmp/vs_Cs_A_2"); write_to_file(vs_B1, "tmp/vs_B_1")
            h5write("eval_eigenval.h5", "old/eigenvalues", [1.0])  # leftover from a previous run is replaced
            Hamster.collect_output_files(["Es", "vs"])
            h5open("eval_eigenval.h5", "r") do f
                @test sort(keys(f)) == ["B", "Cs_A"]
                @test read(f["Cs_A/config_inds"]) == [2, 7]
                @test read(f["Cs_A/eigenvalues"]) == cat(Es_A2, Es_A7, dims=3)
                @test read(f["Cs_A/eigenvectors"]) == cat(vs_A2, vs_A7, dims=4)
                @test read(f["B/config_inds"]) == [1]
                @test read(f["B/eigenvalues"]) == reshape(Es_B1, 4, 3, 1)
                @test read(f["B/eigenvectors"]) == reshape(vs_B1, 4, 4, 3, 1)
            end
            @test !any(file -> endswith(file, ".dat"), readdir())
            @test !isfile("eval_config_inds.h5")
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