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
function write_ham(H, vecs, comm, ind=0; filename="ham.h5", space="k", system="", rank=0, nranks=1)
    for r in 0:nranks-1
        if r == rank
            h5open(filename, "cw") do file
                h_group = ind == 0 ? "H$space" : "H$(space)_$(system)_$ind"
                g = create_group(file, h_group)
                g["vecs"] = space == "r" ? Int.(vecs) : vecs
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
        MPI.Barrier(comm)
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
function read_ham(comm, ind=0; filename="ham.h5", space="k", system="")
    H = nothing
    vecs = nothing
    h5open(filename, "r", comm) do file
        h_group = ind == 0 ? "H$space" : "H$(space)_$(system)_$ind"
        g = file[h_group]
        vecs = read(g["vecs"])
        Nε = read(g[keys(g)[1]]["m"])
        H = get_empty_complex_hamiltonians(Nε, size(vecs, 2), Sparse())

        block_names = sort(filter(x -> x != "vecs", keys(g)))
        for name in block_names
            grp = g[name]
            m = read(grp["m"])
            n = read(grp["n"])
            rowval = read(grp["rowval"])
            colptr = read(grp["colptr"])
            nzval  = read(grp["nzval"])
            H[parse(Int64, name)] = SparseMatrixCSC(m, n, colptr, rowval, nzval)
        end
    end
    return H, vecs
end

read_ham(ind::Integer=0; filename="ham.h5", space="k") = read_ham(MPI.COMM_WORLD, ind; filename=filename, space=space)

const ħ_eVfs = 0.6582119569 # ħ in units of eV·fs
"""
    write_current(bonds, comm, ind=0;
                  ham_file="ham.h5",
                  filename="ham.h5",
                  system="",
                  rank=0,
                  nranks=1)

Compute and write real-space current operators to an HDF5 file.

This function constructs the real-space current matrices from the bond vectors
`bonds` and the real-space Hamiltonian read from `ham_file`. For each lattice
translation `R`, the current operator is computed as

    C(R) = -i / ħ · bonds[R] * H(R),

and written in sparse CSC format to the HDF5 file `filename`.

Each MPI rank writes sequentially to the file, synchronized via barriers, to
avoid concurrent write conflicts. The resulting datasets are stored under a
group named `Cr` (or `Cr_<system>_<ind>` if `ind ≠ 0`), with one subgroup per
lattice translation.

Sparse matrices are stored explicitly via their `rowval`, `colptr`, and Cartesian
components (`xnzval`, `ynzval`, `znzval`) of the nonzero entries.

# Arguments
- `bonds`: Vector of sparse bond matrices indexed by lattice translation, with
  `SVector{3,Float64}` entries.
- `comm`: MPI communicator.
- `ind`: Index selecting the Hamiltonian block to read (default: `0`).

# Keyword Arguments
- `ham_file`: HDF5 file containing the real-space Hamiltonian.
- `filename`: Output HDF5 file to which current operators are written.
- `system`: Optional system label used in the HDF5 group name.
- `rank`: MPI rank of the calling process.
- `nranks`: Total number of MPI ranks participating in the write.

# Notes
- The reduced Planck constant `ħ` is assumed to be given in units of eV·fs.
- The current vectors have units of Å/fs and correspond to velocity matrix elements 
  (current divided by electron charge).
- The function performs no collective MPI I/O; writes are serialized across
  ranks using barriers.
"""
function write_current(bonds, comm, ind=0; ham_file="ham.h5", filename="ham.h5", system="", rank=0, nranks=1)
    Hr, hr_vecs = read_ham(comm, ind, filename=ham_file, space="r", system=system)
    for r in 0:nranks-1
        if r == rank
            h5open(filename, "cw") do file
                h_group = ind == 0 ? "Cr" : "Cr_$(system)_$ind"
                g = create_group(file, h_group)
                g["vecs"] = hr_vecs
                for R in eachindex(bonds)
                    grp = create_group(g, "$R")
                    
                    Cx, Cy, Cz = map(bonds[R]) do bonds_i
                        bs = size(Hr[R], 1) == 2*size(bonds_i, 1) ? apply_spin_basis(bonds_i) : bonds_i
                        elementwise_union_mul(bs, Hr[R], ħ_eVfs)
                    end
                    
                    @assert Cx.rowval == Cy.rowval
                    @assert Cx.colptr == Cy.colptr                  
                    @assert Cx.rowval == Cz.rowval
                    @assert Cx.colptr == Cz.colptr

                    grp["rowval"] = Cx.rowval
                    grp["colptr"] = Cx.colptr
                    grp["xnzval"] = Cx.nzval
                    grp["ynzval"] = Cy.nzval
                    grp["znzval"] = Cz.nzval
                    grp["m"]      = size(Cx, 1)
                    grp["n"]      = size(Cx, 2)
                end
            end
        end
        MPI.Barrier(comm)
    end
end

function elementwise_union_mul(bs::SparseMatrixCSC, Hr::SparseMatrixCSC, ħ_eVfs)
    row_bs, col_bs, _ = findnz(bs)
    row_Hr, col_Hr, _ = findnz(Hr)

    nz_indices = Set{Tuple{Int,Int}}()
    foreach(t -> push!(nz_indices, t), zip(row_bs, col_bs))
    foreach(t -> push!(nz_indices, t), zip(row_Hr, col_Hr))

    i_all = Int[]
    j_all = Int[]
    vals = ComplexF64[]

    for (i, j) in nz_indices
        v_bs = bs[i, j]
        v_Hr = Hr[i, j]
        push!(i_all, i)
        push!(j_all, j)
        push!(vals, (-1im / ħ_eVfs) * v_bs * v_Hr)
    end

    return sparse(i_all, j_all, vals, size(bs, 1), size(bs, 2))
end

"""
    read_current(comm, ind=0;
                 filename="ham.h5",
                 space="r",
                 system="")

Read real-space current operators from an HDF5 file.

This function reads sparse current matrices previously written by
[`write_current`](@ref) from the HDF5 file `filename`. For each lattice
translation `R`, the current operator is reconstructed as a sparse matrix with
`SVector{3,ComplexF64}` entries representing the Cartesian components of the
current.

The current matrices are read from the group `C<space>` (or
`C<space>_<system>_<ind>` if `ind ≠ 0`), along with the associated lattice
translation vectors.

# Arguments
- `comm`: MPI communicator used for collective HDF5 access.
- `ind`: Index selecting the current block to read (default: `0`).

# Keyword Arguments
- `filename`: HDF5 file containing the current operators.
- `space`: Label identifying the representation (e.g. `"r"` for real space).
- `system`: Optional system label used in the HDF5 group name.

# Returns
- `C::Vector{SparseMatrixCSC{SVector{3,ComplexF64},Int64}}`:
  A vector of sparse current matrices, one per lattice translation.
- `vecs::AbstractMatrix{<:Real}`:
  Lattice translation vectors corresponding to the current blocks.

# Notes
- Sparse matrices are reconstructed from their stored CSC components
  (`rowval`, `colptr`, `xnzval`, `ynzval`, `znzval`).
- The function assumes the file layout produced by `write_current`.
"""
function read_current(comm, ind=0; filename="ham.h5", space="r", system="")
    Cx = nothing
    Cy = nothing
    Cz = nothing
    vecs = nothing
    h5open(filename, "r", comm) do file
        @show keys(file)
        h_group = ind == 0 ? "C$space" : "C$(space)_$(system)_$ind"
        g = file[h_group]
        vecs = read(g["vecs"])
        
        Nε = read(g[keys(g)[1]]["m"])
        Cx = SparseMatrixCSC{ComplexF64, Int64}[spzeros(ComplexF64, Nε, Nε) for R in axes(vecs, 2)]
        Cy = SparseMatrixCSC{ComplexF64, Int64}[spzeros(ComplexF64, Nε, Nε) for R in axes(vecs, 2)]
        Cz = SparseMatrixCSC{ComplexF64, Int64}[spzeros(ComplexF64, Nε, Nε) for R in axes(vecs, 2)]

        block_names = sort(filter(x -> x != "vecs", keys(g)))
        for name in block_names
            grp = g[name]
            m = read(grp["m"])
            n = read(grp["n"])
            rowval = read(grp["rowval"])
            colptr = read(grp["colptr"])
            xs = read(grp["xnzval"]); ys = read(grp["ynzval"]); zs = read(grp["znzval"])
            Cx[parse(Int64, name)] = SparseMatrixCSC(m, n, colptr, rowval, xs)
            Cy[parse(Int64, name)] = SparseMatrixCSC(m, n, colptr, rowval, ys)
            Cz[parse(Int64, name)] = SparseMatrixCSC(m, n, colptr, rowval, zs)
        end
    end
    return Cx, Cy, Cz, vecs
end

read_current(ind::Integer=0; filename="ham.h5", space="r") = read_current(MPI.COMM_WORLD, ind; filename=filename, space=space)
