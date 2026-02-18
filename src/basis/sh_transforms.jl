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

# s - orbital 
fs(baseorb::s, θ, φ) = 1.

# p - orbitals
fpx(::px, θ, φ) = cos(φ)*cos(θ)
fpy(::px, θ, φ) = -sin(φ)
fpz(::px, θ, φ) = cos(φ)*sin(θ)

fpx(::py, θ, φ) = sin(φ)*cos(θ)
fpy(::py, θ, φ) = cos(φ)
fpz(::py, θ, φ) = sin(φ)*sin(θ)

fpx(::pz, θ, φ) = -sin(θ)
fpy(::pz, θ, φ) = 0
fpz(::pz, θ, φ) = cos(θ)

fpx(::sp3, θ, φ) = sin(θ)*cos(φ)
fpy(::sp3, θ, φ) = sin(θ)*sin(φ)
fpz(::sp3, θ, φ) = cos(θ)

fpx(::sp3dr2, θ, φ) = sin(θ)*cos(φ)
fpy(::sp3dr2, θ, φ) = sin(θ)*sin(φ)
fpz(::sp3dr2, θ, φ) = cos(θ)

# d - orbitals
fdxy(baseorb::dxy, θ, φ) = cos(φ)*cos(θ)*cos(φ) - sin(φ)*cos(θ)*sin(φ) # 2, -2 to 2, -2
fdxy(baseorb::dyz, θ, φ) = -sin(θ)*cos(φ) # 2, -2 to 2, -1
fdxy(baseorb::dz2, θ, φ) =  0 # 2, -2 to 2, 0
fdxy(baseorb::dxz, θ, φ) = sin(θ)*sin(φ) # 2, -2 to 2, 1
fdxy(baseorb::dx2_y2, θ, φ) = -cos(φ)*cos(θ)*sin(φ) - sin(φ)*cos(θ)*cos(φ) # 2, -2 to 2, 2

fdyz(baseorb::dxy, θ, φ) = -sin(φ)*sin(θ)*sin(φ) + cos(φ)*sin(θ)*cos(φ) # 2, -1 to 2, -2
fdyz(baseorb::dyz, θ, φ) = cos(φ)*cos(θ)# 2, -1 to 2, -1
fdyz(baseorb::dz2, θ, φ) = 0 # 2, -1 to 2, 0
fdyz(baseorb::dxz, θ, φ) =  -sin(φ)*cos(θ)# 2, -1 to 2, 1
fdyz(baseorb::dx2_y2, θ, φ) = -sin(φ)*sin(θ)*cos(φ) - cos(φ)*sin(θ)*sin(φ)# 2, -1 to 2, 2

fdz2(baseorb::dxy, θ, φ) = √3*sin(θ)^2*sin(φ)*cos(φ) # 2, 0 to 2, -2
fdz2(baseorb::dyz, θ, φ) = √3*sin(θ)*cos(θ)*sin(φ) # 2, 0 to 2, -1
fdz2(baseorb::dz2, θ, φ) = (3*cos(θ)^2-1)/2 # 2, 0 to 2, 0
fdz2(baseorb::dxz, θ, φ) = √3*sin(θ)*cos(θ)*cos(φ) # 2, 0 to 2, 1
fdz2(baseorb::dx2_y2, θ, φ) = √3/2*(sin(θ)^2*cos(φ)^2 - sin(θ)^2*sin(φ)^2) # 2, 0 to 2, 2

fdxz(baseorb::dxy, θ, φ) = cos(φ)*cos(θ)*sin(θ)*sin(φ) + sin(φ)*cos(θ)*sin(θ)*cos(φ) # 2, 1 to 2, -2
fdxz(baseorb::dyz, θ, φ) = sin(φ)*cos(θ)*cos(θ) - sin(θ)*sin(θ)*sin(φ) # 2, 1 to 2, -1
fdxz(baseorb::dz2, θ, φ) = -√3*sin(θ)*cos(θ) # 2, 1 to 2, 0
fdxz(baseorb::dxz, θ, φ) = cos(φ)*cos(θ)*cos(θ) - sin(θ)*sin(θ)*cos(φ) # 2, 1 to 2, 1
fdxz(baseorb::dx2_y2, θ, φ) = cos(φ)*cos(θ)*sin(θ)*cos(φ) - sin(φ)*cos(θ)*sin(θ)*sin(φ) # 2, 1 to 2, 2

fdx2_y2(baseorb::dxy, θ, φ) = cos(φ)*cos(θ)*sin(φ)*cos(θ) + sin(φ)*cos(φ) # 2, 2 to 2, -2
fdx2_y2(baseorb::dyz, θ, φ) = -sin(φ)*cos(θ)*sin(θ) # 2, 2 to 2, -1
fdx2_y2(baseorb::dz2, θ, φ) = √3*(sin(θ)^2)/2 # 2, 2 to 2, 0
fdx2_y2(baseorb::dxz, θ, φ) = -cos(φ)*cos(θ)*sin(θ) # 2, 2 to 2, 1
fdx2_y2(baseorb::dx2_y2, θ, φ) = (cos(φ)^2*cos(θ)^2 - sin(φ)^2*cos(θ)^2 - sin(φ)^2 + cos(φ)^2)/2 # 2, 2 to 2, 2

# hybrid orbitals (same as dz2)
fdxy(baseorb::sp3dr2, θ, φ) = √3*sin(θ)^2*sin(φ)*cos(φ)
fdyz(baseorb::sp3dr2, θ, φ) = √3*sin(θ)*cos(θ)*sin(φ)
fdxz(baseorb::sp3dr2, θ, φ) = √3*sin(θ)*cos(θ)*cos(φ)
fdz2(baseorb::sp3dr2, θ, φ) = (3*cos(θ)^2-1)/2
fdx2_y2(baseorb::sp3dr2, θ, φ) = √3/2*(sin(θ)^2*cos(φ)^2 - sin(θ)^2*sin(φ)^2)

# Define normalization constants
const Nspd_s = Float64[1, 0, 0]
const Nspd_p = Float64[0, 3, 0]
const Nspd_d = Float64[0, 0, 5]
const Nspd_sp3 = Float64[1, 3, 0]
const Nspd_sp3dr2 = Float64[1, 3, 5]

Nspd(baseorb::s) = Nspd_s
Nspd(baseorb::px) = Nspd_p
Nspd(baseorb::py) = Nspd_p
Nspd(baseorb::pz) = Nspd_p
Nspd(baseorb::dxy) = Nspd_d
Nspd(baseorb::dxz) = Nspd_d
Nspd(baseorb::dyz) = Nspd_d
Nspd(baseorb::dz2) = Nspd_d
Nspd(baseorb::dx2_y2) = Nspd_d
Nspd(baseorb::sp3) = Nspd_sp3
Nspd(baseorb::sp3dr2) = Nspd_sp3dr2