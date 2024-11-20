"""
    read_hrdat(file::String)

Read a Wannier90 `hr.dat` file and extract the Hamiltonian matrix elements, lattice vectors, and degeneracy values.

# Arguments
- `file::String`: The path to the `hr.dat` file.

# Returns
- A tuple containing:
  - `Hᴿ::Array{ComplexF64, 3}`: The Hamiltonian matrix elements, a 3D array with dimensions (num_wann, num_wann, NR).
  - `Rs::Array{Float64, 2}`: The lattice vectors, a 2D array with dimensions (3, NR).
  - `deg::Vector{Int64}`: The degeneracy values, a vector of length NR.
"""
function read_hrdat(file="wannier90_hr.dat")
    lines = open_and_read(file)

    NR = parse(Int64, lines[3])
    num_wann = parse(Int64, lines[2])
    
    lines = split_lines(lines)

    Rs = zeros(3, NR)
    Hᴿ = get_empty_complex_hamiltonians(num_wann, NR)
    
    # Up to 15 degeneracy values are written per line
    Ldeg = Int(ceil(NR / 15))
    deg = collect(Iterators.flatten([parse.(Int64, lines[k]) for k in 4:3+Ldeg]))
    if length(deg) ≠ NR; throw("Invalid number of degeneracy values found!"); end
    
    Rind = 0
    hr_start = 4+Ldeg
    Rvec = rand(Int64, 3)
    for k in hr_start:length(lines)
        # Check if lattice vector is new
        Rvec_new = parse.(Int64, lines[k][1:3])
        if Rvec_new ≠ Rvec; Rind += 1; Rvec = Rvec_new; Rs[:, Rind] = Rvec; end

        i, j = parse.(Int64, lines[k][4:5])
        Hᴿ[Rind][i, j] = parse(Float64, lines[k][6]) + im*parse(Float64, lines[k][7])
    end
    return Hᴿ, Rs, deg
end