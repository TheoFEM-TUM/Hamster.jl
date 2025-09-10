"""
    write_hr(Hr, Rs; filename="hamster_hr", tol=1e-8, verbose=1)

Writes Hamiltonian data `Hr` and corresponding lattice vectors `Rs` to a file.

# Arguments:
- `Hr`: A vector of matrices where each `Hr[R]` contains the Hamiltonian elements corresponding to the lattice vector `Rs[:, R]`.
- `Rs`: A 2D array where each column is a lattice vector corresponding to the Hamiltonian elements in `Hr`. The array has dimensions `(3, NR)`, representing the lattice vectors in 3D space.
- `filename::String="hamster_hr"`: The base name of the output file. The function appends `.dat` to this base name to create the final output file name.
- `tol::Float64=1e-8`: A tolerance threshold for determining whether a Hamiltonian element should be considered non-zero and written to the file. Elements with absolute values below this threshold are omitted from the output.
- `verbose::Int64`: Sets the verbosity. `verbose=0` deactivates print statements.

# Behavior:
- The function writes the Hamiltonian data to a file named `filename.dat`.
- The first line of the file contains the size of the Hamiltonian matrix (`Nε`).
- The second line contains the number of lattice vectors (`NR`).
- The function writes only the non-zero elements of `Hr` (as determined by `tol`) to the file, along with their corresponding lattice vector indices and matrix indices.
- The output file format aligns the numeric values to fixed-width columns for readability.
"""
function write_hr(Hr, Rs; filename="hamster_hr", tol=1e-10, verbose=1)
    hr_dat = open(filename*".dat", "w")
    Nε = size(Hr[1], 1)
    NR = size(Rs, 2)
    println(hr_dat, string(Nε))
    println(hr_dat, string(NR))

    LR = 4; Lij = 6; LHr = 25
    Ntot = prod(size(Hr))
    Nsav = 0
    time = @elapsed @views for R in 1:NR, j in 1:Nε, i in 1:Nε
        Hr_ij = Hr[R][i, j]
        Rx, Ry, Rz = Int.(Rs[:, R])
        Lx = length(string(Rx)); Ly = length(string(Ry)); Lz = length(string(Rz))
        Li = length(string(i)); Lj = length(string(j)); LHrij = length(string(Hr_ij))
        if abs(Hr_ij) > tol
            println(hr_dat, "  ", 
                        " "^(LR-Lx)*string(Rx), 
                        " "^(LR-Ly)*string(Ry), 
                        " "^(LR-Lz)*string(Rz),
                        " "^(Lij-Li)*string(i),
                        " "^(Lij-Lj)*string(j),
                        " "^(LHr-LHrij)*string(Hr_ij)
                        )
            Nsav += 1
        end
    end
    if verbose ≥ 1
        println("Hr sparsity: ", (Ntot - Nsav) / Ntot)
        println("Time to write: $time s")
    end
    close(hr_dat)
end

"""
    read_hr(filename="hamster_hr.dat"; sp_mode=false, verbose=1)

Reads Hamiltonian data and corresponding lattice vectors from a file, reconstructing the Hamiltonian matrices and lattice vectors.

# Arguments:
- `filename::String="hamster_hr.dat"`: The name of the file containing the Hamiltonian data. The default file name is `"hamster_hr.dat"`.
- `sp_mode::Bool=false`: If `true`, enables sparse matrix storage mode for the Hamiltonians. This is useful for large, sparse Hamiltonians to save memory.
- `verbose::Int64`: Sets the verbosity. `verbose=0` deactivates print statements.

# Returns:
- `Hr`: A vector of 2D arrays where each element `Hr[R]` is a Hamiltonian matrix corresponding to the lattice vector `Rs[:, R]`. Each `Hr[R]` is a 2D array of size `(Nε, Nε)`, where `Nε` is the size of the Hamiltonian matrices.
- `Rs`: A 2D array where each column is a lattice vector corresponding to the Hamiltonians in `Hr`. The array has dimensions `(3, NR)`, representing the lattice vectors in 3D space.

# Behavior:
- The function reads the Hamiltonian data from the specified file.
- The first line of the file specifies the size of the Hamiltonian matrix (`Nε`).
- The second line specifies the number of lattice vectors (`NR`).
- The function reads the subsequent lines to extract the Hamiltonian elements and their corresponding lattice vectors.
- If `sp_mode` is `true`, the Hamiltonians are stored in a sparse format. Otherwise, a dense format is used.
"""
function read_hr(filename="hamster_hr.dat"; sp_mode=false, verbose=1)
    lines = open_and_read(filename)
    lines = split_lines(lines)
    Nε = parse(Int64, lines[1][1])
    NR = parse(Int64, lines[2][1])
    mode = sp_mode ? Sparse() : Dense()
    Hr = get_empty_real_hamiltonians(Nε, NR, mode)
    Rs = zeros(Int64, 3, NR)
    R = 0
    old_R = zeros(3)
    time = @elapsed for (l, line) in enumerate(lines[3:end])
        new_R = parse.(Int64, line[1:3])
        if new_R ≠ old_R || l == 1; R += 1; Rs[:, R] .= new_R; end
        i, j = parse.(Int64, line[4:5])
        Hr[R][i, j] = parse(Float64, line[6])
        old_R .= new_R
    end
    if verbose ≥ 1; println("Time to read: $time s"); end
    return Hr, Rs
end

"""
    write_ham(H, vecs, comm, ind=0; filename="ham.h5", space="k")

Write a Hamiltonian, represented as a vector of sparse matrices, into an HDF5 file
in a parallel MPI setting.

# Arguments
- `H::Vector{SparseMatrixCSC}`: Vector of sparse matrices that form the Hamiltonian blocks.
- `vecs::AbstractArray`: Array of vectors with one vector per block (k or R vectors).
- `comm::MPI.Comm`: MPI communicator used to open the HDF5 file in parallel mode.
- `ind::Int=0`: Optional index label (e.g., for each atomic configuration).

# File structure
The Hamiltonian is stored under a group named:
- `"H\$space"` if `ind == 0`
- `"H\${space}_\$ind"` otherwise

Inside this group:
- `"vecs"`: dataset containing the supplied `vecs` (k or R vectors).
- Subgroups `"1"`, `"2"`, …, one for each block in `H`, containing:
  - `"rowval"`, `"colptr"`, `"nzval"`: the CSC representation of the sparse matrix.
  - `"m"`, `"n"`: matrix dimensions.
"""
function write_ham(H, vecs, comm, ind=0; filename="ham.h5", space="k")
    h5open(filename, "cw", comm) do file
        h_group = ind == 0 ? "H$space" : "H$(space)_$ind"
        g = create_group(file, h_group)
        g["vecs"] = vecs
        for (i, mat) in enumerate(H)
            smat = issparse(mat) ? mat : sparse(mat)
            grp = create_group(g, "$i")
            grp["rowval"] = smat.rowval
            grp["colptr"] = smat.colptr
            grp["nzval"]  = smat.nzval
            grp["m"]      = size(smat, 1)
            grp["n"]      = size(smat, 2)
        end
    end
end

"""
    read_ham(comm, ind=0; filename="ham.h5", space="k")

Read a Hamiltonian and associated vectors from an HDF5 file previously written with `write_ham`.

# Arguments
- `comm::MPI.Comm`: MPI communicator used to open the HDF5 file in parallel mode.
- `ind::Int=0`: Optional index to identify which Hamiltonian group to read.

# Keyword Arguments
- `filename::AbstractString="ham.h5"`: Name of the HDF5 file to read from.
- `space::AbstractString="k"`: Label used in the Hamiltonian group name (e.g., `"Hk"` or `"Hk_1"`).

# Returns
- `H::Vector{SparseMatrixCSC}`: Vector of Hamiltonian blocks reconstructed as sparse matrices.
- `vecs::Array`: The stored array of vectors associated with the Hamiltonian.
"""
function read_ham(comm, ind=0; filename="ham.h5", space="k")
    H = SparseMatrixCSC[]
    vecs = nothing
    h5open(filename, "r", comm) do file
        h_group = ind == 0 ? "H$space" : "H$(space)_$ind"
        g = file[h_group]
        vecs = read(g["vecs"])
        # Ensure blocks are read in order "1", "2", ...
        block_names = sort(filter(x -> x != "vecs", keys(g)))
        for name in block_names
            grp = g[name]
            m = read(grp["m"])
            n = read(grp["n"])
            rowval = read(grp["rowval"])
            colptr = read(grp["colptr"])
            nzval  = read(grp["nzval"])
            push!(H, SparseMatrixCSC(m, n, colptr, rowval, nzval))
        end
    end
    return H, vecs
end