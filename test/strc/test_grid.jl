@testset "Push Points" begin
    
    # Test 1: Adding a new grid point to an empty dictionary
    grid_dict = Dict{SVector{3, Int64}, Vector{Int64}}()
    grid_point = SVector{3, Int64}(1, 1, 1)
    point_info = 1
    Hamster.push_grid_point!(grid_dict, grid_point, point_info)
    @test haskey(grid_dict, grid_point)
    @test grid_dict[grid_point] == [point_info]
    
    # Test 2: Adding a point to an existing grid point in the dictionary
    point_info_2 = 2
    Hamster.push_grid_point!(grid_dict, grid_point, point_info_2)
    @test grid_dict[grid_point] == [point_info, point_info_2]

    # Test 3: Adding a new grid point to a non-empty dictionary
    new_grid_point = SVector{3, Int64}(2, 2, 2)
    point_info_3 = 3
    Hamster.push_grid_point!(grid_dict, new_grid_point, point_info_3)
    @test haskey(grid_dict, new_grid_point)
    @test grid_dict[new_grid_point] == [point_info_3]
    
    # Test 4: Check if the function correctly handles different types of `point_info`
    another_grid_dict = Dict{SVector{3, Int64}, Vector{Tuple{Int64, Int64}}}()
    complex_grid_point = SVector{3, Int64}(3, 3, 3)
    complex_point_info = (5, 6)
    Hamster.push_grid_point!(another_grid_dict, complex_grid_point, complex_point_info)
    @test haskey(another_grid_dict, complex_grid_point)
    @test another_grid_dict[complex_grid_point] == [complex_point_info]
end

@testset "Grid Points" begin

    # Test 1: Basic functionality with simple input
    @test Hamster.get_grid_point(SVector{3}(1.5, 2.7, 3.9), 1.0) == SVector{3, Int64}(1, 2, 3)
    
    # Test 2: Input that falls exactly on the grid boundaries
    @test Hamster.get_grid_point(SVector{3}(2.0, 2.0, 2.0), 1.0) == SVector{3, Int64}(2, 2, 2)

    # Test 3: Larger grid size
    @test Hamster.get_grid_point(SVector{3}(3.9, 4.1, 5.5), 2.0) == SVector{3, Int64}(1, 2, 2)
    
    # Test 4: Non-uniform grid size
    @test Hamster.get_grid_point(SVector{3}(10.5, 20.7, 30.9), 10.0) == SVector{3, Int64}(1, 2, 3)
    
    # Test 5: Small grid size with negative coordinates
    @test Hamster.get_grid_point(SVector{3}(-1.2, -2.8, -3.3), 0.5) == SVector{3, Int64}(-3, -6, -7)

    # Test 6: Grid size less than 1
    @test Hamster.get_grid_point(SVector{3}(0.15, 0.27, 0.39), 0.1) == SVector{3, Int64}(1, 2, 3)
    
    # Test 7: Zero position vector
    @test Hamster.get_grid_point(SVector{3}(0.0, 0.0, 0.0), 1.0) == SVector{3, Int64}(0, 0, 0)
end

@testset "PointGrid" begin
    rs = 2 .* rand(3, 16) .- 1
    Ts = 2 .* rand(3, 5) .- 1

    point_grid = Hamster.PointGrid(rs, Ts, grid_size=0.5)

    # Test 1: dict0 should have 16 unique points
    @test sum(length.(unique(unique.(values(point_grid.dict0))))) == size(rs, 2)

    # Test 2: dictR should have 16*5 unique points
    @test sum(length.(unique(unique.(values(point_grid.dictR))))) == size(rs, 2)*size(Ts, 2)

    #Test 3: Check that fields are correct
    @test length(keys(point_grid.dictR)) == point_grid.num_points
    @test point_grid.grid_size == 0.5

    # Test 4: Check behavior with Config
    rs = 2 .* rand(3, 5) .- 1
    Ts = 2 .* rand(3, 3) .- 1
    conf = Hamster.get_empty_config()
    point_grid = Hamster.PointGrid(rs, Ts, conf)
    @test point_grid.grid_size == Hamster.get_grid_size(conf)

    # Rcut is used as grid_size as default
    Hamster.set_value!(conf, "rcut", "0.2")
    point_grid_2 = Hamster.PointGrid(rs, Ts, conf)
    @test point_grid_2.grid_size == 0.2

    # If grid_size is set, its value is used instead
    Hamster.set_value!(conf, "grid_size", "0.1")
    point_grid_3 = Hamster.PointGrid(rs, Ts, conf)
    @test point_grid_3.grid_size == 0.1
end