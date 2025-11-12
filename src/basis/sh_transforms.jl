"""
    baseorb logic

The baseorb of px, py, pz should be the same since since one can transform one into 
the other by rotation.
"""
function get_base_orb(base_1::A1, base_2::A2; bondswap=false) where {A1,A2<:Angular}
    if bondswap
        return (get_base_orb(base_2), get_base_orb(base_1))
    else
        return (get_base_orb(base_1), get_base_orb(base_2))
    end
end
get_base_orb(orb::Angular) = orb
get_base_orb(orb::px) = porb()
get_base_orb(orb::py) = porb()
get_base_orb(orb::pz) = porb()

get_base_orb(orb::pxdx2) = prdr2()
get_base_orb(orb::pydy2) = prdr2()
get_base_orb(orb::pzdz2) = prdr2()

# s - orbital 
fs(θ, φ, baseorb::Angular) = 1.

# p - orbitals
fpx(θ, φ) = sin(θ)*cos(φ)
fpy(θ, φ) = sin(θ)*sin(φ)
fpz(θ, φ) = cos(θ)

# d - orbitals
fdxy(baseorb::dxy, θ, φ) = cos(φ)*cos(θ)*cos(φ) - sin(φ)*cos(θ)*sin(φ) # 2, -2 to 2, -2
fdyz(baseorb::dxy, θ, φ) = -sin(θ)*cos(φ) # 2, -2 to 2, -1
fdz2(baseorb::dxy, θ, φ) =  0 # 2, -2 to 2, 0
fdxz(baseorb::dxy, θ, φ) = sin(θ)*sin(φ) # 2, -2 to 2, 1
fdx2_y2(baseorb::dxy, θ, φ) = -cos(φ)*cos(θ)*sin(φ) - sin(φ)*cos(θ)*cos(φ) # 2, -2 to 2, 2

fdxy(baseorb::dyz, θ, φ) = -sin(φ)*sin(θ)*sin(φ) + cos(φ)*sin(θ)*cos(φ) # 2, -1 to 2, -2
fdyz(baseorb::dyz, θ, φ) = cos(φ)*cos(θ)# 2, -1 to 2, -1
fdz2(baseorb::dyz, θ, φ) = 0 # 2, -1 to 2, 0
fdxz(baseorb::dyz, θ, φ) =  -sin(φ)*cos(θ)# 2, -1 to 2, 1
fdx2_y2(baseorb::dyz, θ, φ) = -sin(φ)*sin(θ)*cos(φ) - cos(φ)*sin(θ)*sin(φ)# 2, -1 to 2, 2

fdxy(baseorb::dz2, θ, φ) = √3*sin(θ)^2*sin(φ)*cos(φ) # 2, 0 to 2, -2
fdyz(baseorb::dz2, θ, φ) = √3*sin(θ)*cos(θ)*sin(φ) # 2, 0 to 2, -1
fdz2(baseorb::dz2, θ, φ) = (3*cos(θ)^2-1)/2 # 2, 0 to 2, 0
fdxz(baseorb::dz2, θ, φ) = √3*sin(θ)*cos(θ)*cos(φ) # 2, 0 to 2, 1
fdx2_y2(baseorb::dz2, θ, φ) = √3/2*(sin(θ)^2*cos(φ)^2 - sin(θ)^2*sin(φ)^2) # 2, 0 to 2, 2

fdxy(baseorb::dxz, θ, φ) = cos(φ)*cos(θ)*sin(θ)*sin(φ) + sin(φ)*cos(θ)*sin(θ)*cos(φ) # 2, 1 to 2, -2
fdyz(baseorb::dxz, θ, φ) = sin(φ)*cos(θ)*cos(θ) - sin(θ)*sin(θ)*sin(φ) # 2, 1 to 2, -1
fdz2(baseorb::dxz, θ, φ) = -√3*sin(θ)*cos(θ) # 2, 1 to 2, 0
fdxz(baseorb::dxz, θ, φ) = cos(φ)*cos(θ)*cos(θ) - sin(θ)*sin(θ)*cos(φ) # 2, 1 to 2, 1
fdx2_y2(baseorb::dxz, θ, φ) = cos(φ)*cos(θ)*sin(θ)*cos(φ) - sin(φ)*cos(θ)*sin(θ)*sin(φ) # 2, 1 to 2, 2

fdxy(baseorb::dx2_y2, θ, φ) = cos(φ)*cos(θ)*sin(φ)*cos(θ) + sin(φ)*cos(φ) # 2, 2 to 2, -2
fdyz(baseorb::dx2_y2, θ, φ) = -sin(φ)*cos(θ)*sin(θ) # 2, 2 to 2, -1
fdz2(baseorb::dx2_y2, θ, φ) = √3*(sin(θ)^2)/2 # 2, 2 to 2, 0
fdxz(baseorb::dx2_y2, θ, φ) = -cos(φ)*cos(θ)*sin(θ) # 2, 2 to 2, 1
fdx2_y2(baseorb::dx2_y2, θ, φ) = (cos(φ)^2*cos(θ)^2 - sin(φ)^2*cos(θ)^2 - sin(φ)^2 + cos(φ)^2)/2 # 2, 2 to 2, 2

# hybrid orbitals (same as dz2)
fdxy(baseorb::sp3dr2, θ, φ) = √3*sin(θ)^2*sin(φ)*cos(φ)
fdyz(baseorb::sp3dr2, θ, φ) = √3*sin(θ)*cos(θ)*sin(φ)
fdxz(baseorb::sp3dr2, θ, φ) = √3*sin(θ)*cos(θ)*cos(φ)
fdz2(baseorb::sp3dr2, θ, φ) = (3*cos(θ)^2-1)/2
fdx2_y2(baseorb::sp3dr2, θ, φ) = √3/2*(sin(θ)^2*cos(φ)^2 - sin(θ)^2*sin(φ)^2)

fdxy(baseorb::prdr2, θ, φ) = √3*sin(θ)^2*sin(φ)*cos(φ)
fdyz(baseorb::prdr2, θ, φ) = √3*sin(θ)*cos(θ)*sin(φ)
fdxz(baseorb::prdr2, θ, φ) = √3*sin(θ)*cos(θ)*cos(φ)
fdz2(baseorb::prdr2, θ, φ) = (3*cos(θ)^2-1)/2
fdx2_y2(baseorb::prdr2, θ, φ) = √3/2*(sin(θ)^2*cos(φ)^2 - sin(θ)^2*sin(φ)^2)

# Define normalization constants
Nspd(baseorb::Angular) = Float64[1, 1, 1]
Nspd(baseorb::s) = Float64[1, 0, 0]
Nspd(baseorb::porb) = Float64[0, 3, 0]
Nspd(baseorb::px) = Float64[0, 3, 0]
Nspd(baseorb::py) = Float64[0, 3, 0]
Nspd(baseorb::pz) = Float64[0, 3, 0]
Nspd(baseorb::dorb) = Float64[0, 0, 5]
Nspd(baseorb::dxy) = Float64[0, 0, 5]
Nspd(baseorb::dxz) = Float64[0, 0, 5]
Nspd(baseorb::dyz) = Float64[0, 0, 5]
Nspd(baseorb::dz2) = Float64[0, 0, 5]
Nspd(baseorb::dx2_y2) = Float64[0, 0, 5]
const Nspd_sp3 = Float64[1, 3, 0]
Nspd(baseorb::sp3) = Nspd_sp3
const Nspd_sp3dr2 = Float64[1, 3, 5]
Nspd(baseorb::sp3dr2) = Nspd_sp3dr2
Nspd(baseorb::prdr2) = Float64[0, 3, 5]