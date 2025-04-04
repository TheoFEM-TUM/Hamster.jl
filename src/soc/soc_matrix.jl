"""
    get_soc_matrices(strc::Structure) -> Vector{BlockDiagonal}

Precompute the spin-orbit coupling (SOC) matrices for a given structure.
"""
function get_soc_matrices(strc::Structure, basis::Basis, conf=get_empty_config(); verbosity=get_verbosity(conf))

    # Extract axes for all atoms from the structure
    atom_axes_list = get_axes_from_orbitals(basis.orbitals)
    hybridisation = Dict(-2 => "sp3dr2", -1 => "sp3")

    # Dictionary mapping orbital type to SOC function
    soc_functions = Dict(
        -2 => get_Msoc_ho, # For sp3dr2 orbitals
        -1 => get_Msoc_ho, # For sp3 orbitals
        0 => get_Msoc_s,    # For s orbitals
        1 => get_Msoc_p,    # For p orbitals
        2 => get_Msoc_d,    # For d orbitals
        # ... other mappings as necessary ...
    )

    # Array to store SOC matrices for each ion
    soc_matrices = Matrix{ComplexF64}[] # one atom, one Msoc
    soc_order = Vector{String}[] # one atom, one orb_order
    # Iterate over each ion
    for (iion, ion) in enumerate(strc.ions)
        ls = unique([orb.type.l for orb in basis.orbitals[iion]])
        soc_blocks = Matrix{ComplexF64}[] # skipped/empty for hybrid
        orb_order = String[]
        if ls[1] < 0 # Hybrid
            # Get the correct orientation axes and calculate Msoc_ho for the current atom
            axes = atom_axes_list[iion]
            push!(soc_matrices, get_Msoc_ho(axes, mode=hybridisation[ls[1]])) # TODO

            # Define the basis order that expresses Msoc
            if hybridisation[ls[1]] == "sp3"
                orb_order = ["sp3₁↑", "sp3₁↓", "sp3₂↑", "sp3₂↓", "sp3₃↑", "sp3₃↓", "sp3₄↑", "sp3₄↓"]
            elseif hybridisation[ls[1]] == "sp3dr2"
                orb_order = ["sp3d2₁↑", "sp3d2₁↓", "sp3d2₂↑", "sp3d2₂↓", "sp3d2₃↑", "sp3d2₃↓", "sp3d2₄↑", "sp3d2₄↓"]
            end
            push!(soc_order, orb_order)
        else
            for l in ls # AOs
                ms = [orb.type.m for orb in basis.orbitals[iion] if orb.type.l == l]
                spatial_dims = [lm_to_orbital_map[(l, m)] for m in ms if (l, m) in keys(lm_to_orbital_map)]
                spin_dims = ["↑", "↓"]
                basis_order = [sd * sp for sd in spatial_dims for sp in spin_dims]
                if l in keys(soc_functions); push!(soc_blocks, soc_functions[l](basis_order)); end
                append!(orb_order, basis_order)
            end
            # Allocate different l-matrices into one block-diagonal matrix
            ion_soc_matrix = Matrix(BlockDiagonal(soc_blocks))
            push!(soc_matrices, ion_soc_matrix)
            push!(soc_order, orb_order)
        end
    end
    if verbosity > 1; @show soc_order; end
    return soc_matrices
end

"""
    Ms, Mp, Md, Mtrans

Constants representing transformation matrices for different types of atomic orbitals.

- `Ms`: Transformation matrix for `s` orbitals. As `s` orbitals are spherically symmetrical, 
  the transformation matrix is a 1x1 identity matrix.

- `Mp`: Transformation matrix for `p` orbitals (pz, px, py). This 3x3 matrix is used to convert 
  between the |lm, s> basis and the cartesian coordinate basis (pz, px, py).

- `Md`: Transformation matrix for `d` orbitals (dz2, dxz, dyz, dx2-y2, dxy). This 5x5 matrix 
  is used to convert between the |lm, s> basis and the real spherical harmonics basis commonly
  used to describe `d` orbitals.

# these orbitals are ordered in the default order in Wannier90.
"""

Ms = [1]
Mp = [ 0.0         1.0         0.0;
       1/sqrt(2)   0.0        -1/sqrt(2);
       im/sqrt(2)  0.0         im/sqrt(2)]

Md = [ 0.0         0.0         1.0         0.0         0.0;
       0.0         1/sqrt(2)   0.0        -1/sqrt(2)   0.0;
       0.0         im/sqrt(2)  0.0         im/sqrt(2)  0.0;
       1/sqrt(2)   0.0         0.0         0.0         1/sqrt(2);
       im/sqrt(2)  0.0         0.0         0.0        -im/sqrt(2)]
"""
- `Mtrans`: Dictionary mapping the type of orbital ("s", "p", "d") to its corresponding 
  transformation matrix (Ms, Mp, Md).
"""
const Mtrans = Dict("s" => Ms, "p" => Mp, "d" => Md)


"""
    trans_lm_spatial(orb::String, Msoc::Array{T, 2}) where T -> Array{T, 2}

Transform the spin-orbit coupling matrix `Msoc` from the spin basis to the spatial basis for the given orbital type `orb`.

# Arguments
- `orb`: Orbital type as a string ("s", "p", or "d").
- `Msoc`: Matrix representing the spin-orbit coupling in the spin basis.

# Returns
- The transformed spin-orbit coupling matrix in the spatial basis.
"""
function trans_lm_spatial(orb::String, Msoc::Array{T, 2}) where T
    trans = transpose(kron(Mtrans[orb], Matrix(I, 2, 2)))
    Msoc_spatial = trans' * Msoc * trans
    return Msoc_spatial
end

"""
    gen_permutation_matrix(initial_order::Vector{String}, desired_order::Vector{String}) -> Array{Int, 2}

Generate a permutation matrix `P` that will reorder elements from `initial_order` to `desired_order`.

# Arguments
- `initial_order`: A vector of strings representing the initial order of elements.
- `desired_order`: A vector of strings representing the desired order of elements.

# Returns
- The permutation matrix `P` that reorders `initial_order` to `desired_order`.
"""
function gen_permutation_matrix(initial_order::Vector{String}, desired_order::Vector{String})
    n = length(initial_order)
    P = zeros(Int, n, n)
    desired_dict = Dict(desired_order[i] => i for i in 1:n)
    
    for i in 1:n
        j = get(desired_dict, initial_order[i], nothing)
        if j !== nothing
            P[j, i] = 1
        end
    end
    return P
end

"""
    get_Msoc_s() -> zeros(Complex{Float64}, 2, 2)

Calculate the spin-orbit coupling matrix for the s orbitals / a zero matrix.

# Returns
- The spin-orbit coupling matrix `Msoc_s` for s orbitals.
"""
get_Msoc_s(output_basis_order=["s↑", "s↓"]) = zeros(Complex{Float64}, 2, 2)

"""
    get_Msoc_p(output_basis_order::Vector{String} = ["pz↑", "px↑", "py↑", "pz↓", "px↓", "py↓"]) -> Matrix{Complex{Float64}}

Calculate the spin-orbit coupling matrix for `p` orbitals. The function constructs the matrix in the |lm,s> basis,
transforms it to the spatial basis for `p` orbitals, applies a permutation to match the specified output basis order,
and then applies numerical tolerance to trim small values.

# Arguments
- `output_basis_order`: A vector of strings that defines the order of the orbital and spin basis in the output matrix.
                        The default order is the one given in the thesis.

# Returns
- A matrix representing the spin-orbit coupling in the spatial basis with the specified output order.
"""
function get_Msoc_p(output_basis_order=["pz↑", "px↑", "py↑", "pz↓", "px↓", "py↓"])
    basis = get_lms_basis("p")
    Msoc_lm = get_matrix_lmbasis(basis)
    Msoc_spatial = trans_lm_spatial("p", Msoc_lm)
    # for hybrid: ["px↑", "px↓", "py↑", "py↓", "pz↑", "pz↓"]
    P = gen_permutation_matrix(["pz↓", "pz↑", "px↓", "px↑", "py↓", "py↑"], output_basis_order)
    Msoc_spatial_perm = P * Msoc_spatial * P'
    return thresholding.(Msoc_spatial_perm)
end

"""
    get_Msoc_d(output_basis_order::Vector{String} = ["dz2↑", "dxz↑", "dyz↑", "dx2-y2↑", "dxy↑", "dz2↓", "dxz↓", "dyz↓", "dx2-y2↓", "dxy↓"]) -> Matrix{Complex{Float64}}

Construct the spin-orbit coupling matrix for `d` orbitals in the spatial basis. This function starts by generating the
basis in the |lm,s> representation, calculates the spin-orbit coupling matrix in this basis, and then transforms it to
the spatial basis. A permutation is applied to align with the specified output basis order, and numerical tolerance is
used to set small values effectively to zero.

# Arguments
- `output_basis_order`: The order of orbitals and spins in the output matrix, provided as an array of strings.
                        The default order is the one given in the thesis.

# Returns
- The spin-orbit coupling matrix for `d` orbitals, ordered according to `output_basis_order`.
"""
function get_Msoc_d(output_basis_order=["dz2↑", "dxz↑", "dyz↑", "dx2-y2↑", "dxy↑", "dz2↓", "dxz↓", "dyz↓", "dx2-y2↓", "dxy↓"])
    basis = get_lms_basis("d")
    Msoc_lm = get_matrix_lmbasis(basis)
    Msoc_spatial = trans_lm_spatial("d", Msoc_lm)
    # for hybrid: ["dxy↑", "dxy↓", "dxz↑", "dxz↓", "dyz↑", "dyz↓", "dz2↑", "dz2↓", "dx2_y2↑", "dx2_y2↓"]
    P = gen_permutation_matrix(["dz2↓", "dz2↑", "dxz↓", "dxz↑", "dyz↓", "dyz↑", "dx2-y2↓", "dx2-y2↑", "dxy↓", "dxy↑"], output_basis_order)
    Msoc_spatial_perm = P * Msoc_spatial * P'
    return thresholding.(Msoc_spatial_perm)
end

"""
    get_Msoc_ho(axes; mode="sp3dr2") -> BlockDiagonal

Calculate the spin-orbit coupling (SOC) matrix for a given atomic axes and hybridization mode.

# Arguments
- `axes::Matrix{Float64}`: The orientation axes of the current atom.
- `mode::String`: The hybridization mode, either "sp3" or "sp3dr2" (default: "sp3dr2").

# Returns
- `BlockDiagonal`: The SOC matrix in the hybridized basis.
"""
function get_Msoc_ho(axes; mode="sp3dr2")

    # Define the order for p and d orbitals
    p_order = ["px↑", "px↓", "py↑", "py↓", "pz↑", "pz↓"]
    d_order = ["dxy↑", "dxy↓", "dxz↑", "dxz↓", "dyz↑", "dyz↓", "dz2↑", "dz2↓", "dx2-y2↑", "dx2-y2↓"]

    # Get the spin-orbit coupling matrices
    Msoc_s = get_Msoc_s()
    Msoc_p = get_Msoc_p(p_order)
    Msoc_d = get_Msoc_d(d_order)
    
    # Create the block diagonal SOC matrix
    Msoc_ao = mode == "sp3" ? BlockDiagonal([Msoc_s, Msoc_p]) : mode == "sp3dr2" ? BlockDiagonal([Msoc_s, Msoc_p, Msoc_d]) : error("Invalid mode")
    Ispin = Array{ComplexF64}(I, 2, 2)

    # Compute hybrid coefficient matrix for the current atom
    T_atom = get_hybrid_coefficient_matrix(axes, mode=mode)
    # Double the Hilbert space (↑↓↑↓ convention)
    C_atom = kron(T_atom', Ispin)
    # Transform the SOC matrix using the hybrid coefficients
    # @show size(T_atom), size(C_atom'), size(T_atom), size(C_atom), size(C_atom)
    Msoc_ho = C_atom' * Msoc_ao * C_atom
    return thresholding.(Msoc_ho)
end

"""
    get_sp3_expansion_coefficients(θ, φ; Nspd=[1, 3, 0]) -> Vector{Float64}

Calculate the sp3 expansion coefficients for given angles θ and φ.

# Arguments
- `θ::Float64`: Polar angle.
- `φ::Float64`: Azimuthal angle.
- `Nspd::Vector{Int}`: Relative proportions of s, p, and d orbitals (default: [1, 3, 0]).

# Returns
- `Vector{Float64}`: The expansion coefficients in the order: s, px, py, pz.
"""
function get_sp3_expansion_coefficients(θ, φ; Nspd=[1, 3, 0])
    Ns = √(Nspd[1]/sum(Nspd)); Np = √(Nspd[2]/sum(Nspd))
    # The order is: s, px, py, pz
    return [Ns, Np*sin(θ)*cos(φ), Np*sin(θ)*sin(φ), Np*cos(θ)]
end

function get_dr2_expansion_coefficients(θ, φ; Nspd=[1, 3, 5])
    Nd = √(Nspd[3]/sum(Nspd))
    # The order is: dxy, dxz, dyz, dz2, dx2_y2
    return Nd .* [√3*sin(θ)^2*sin(φ)*cos(φ), √3*sin(θ)*cos(θ)*cos(φ), √3*sin(θ)*cos(θ)*sin(φ), (3*cos(θ)^2-1)/2, √3/2*(sin(θ)^2*cos(φ)^2 - sin(θ)^2*sin(φ)^2)]
end

get_sp3dr2_expansion_coefficients(θ, φ; Nspd=[1, 3, 5]) = vcat(get_sp3_expansion_coefficients(θ, φ, Nspd=Nspd), get_dr2_expansion_coefficients(θ, φ, Nspd=Nspd))

"""
    get_hybrid_coefficient_matrix(axes; mode="sp3dr2") -> Matrix{ComplexF64}

Calculate the hybrid coefficient matrix for given axes and mode.

# Arguments
- `axes::Matrix{Float64}`: The axes for the atom.
- `mode::String`: The hybridization mode, either "sp3" or "sp3dr2" (default: "sp3dr2").

# Returns
- `Matrix{ComplexF64}`: The hybrid coefficient matrix.
"""
function get_hybrid_coefficient_matrix(axes; mode="sp3dr2")
    Norb = size(axes, 2)
    Nao = mode == "sp3dr2" ? 9 : 4
    Thyb = zeros(Norb, Nao)
    @views for j in 1:Norb
        _, θj, φj = transform_to_spherical(axes[:, j])
        if mode == "sp3dr2"
            Thyb[j, :] .= get_sp3dr2_expansion_coefficients(θj, φj)
        elseif mode == "sp3"
            Thyb[j, :] .= get_sp3_expansion_coefficients(θj, φj)
        end
    end
    return Thyb
end