"""
    baseorb logic

The baseorb of px, py, pz should be the same since since one can transform one into 
the other by rotation.
"""
function get_base_orb(base::Tuple{A1, A2}) where {A1,A2<:Angular}
    base = get_base_orb.(base)
    #if base[1].l > base[2].l && base[1].l ≥ 0 && base[2].l ≥ 0; return (base[2], base[1])
    #else; return base; end
    return base
end
get_base_orb(orb::Angular) = orb
get_base_orb(orb::px) = porb()
get_base_orb(orb::py) = porb()
get_base_orb(orb::pz) = porb()

get_base_orb(orb::pxdx2) = prdr2()
get_base_orb(orb::pydy2) = prdr2()
get_base_orb(orb::pzdz2) = prdr2()


"""
    Define rotation coefficients.
"""
fs(θ, φ, baseorb::Angular) = 1.

fpx(θ, φ) = sin(θ)*cos(φ)
fpy(θ, φ) = sin(θ)*sin(φ)
fpz(θ, φ) = cos(θ)

fdxy(baseorb::dz2, θ, φ) = √3*sin(θ)^2*sin(φ)*cos(φ)
fdyz(baseorb::dz2, θ, φ) = √3*sin(θ)*cos(θ)*sin(φ)
fdxz(baseorb::dz2, θ, φ) = √3*sin(θ)*cos(θ)*cos(φ)
fdz2(baseorb::dz2, θ, φ) = (3*cos(θ)^2-1)/2
fdx2_y2(baseorb::dz2, θ, φ) = √3/2*(sin(θ)^2*cos(φ)^2 - sin(θ)^2*sin(φ)^2)

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
Nspd(baseorb::dorb) = Float64[0, 0, 5]
const Nspd_sp3 = Float64[1, 3, 0]
Nspd(baseorb::sp3) = Nspd_sp3
const Nspd_sp3dr2 = Float64[1, 3, 5]
Nspd(baseorb::sp3dr2) = Nspd_sp3dr2
Nspd(baseorb::prdr2) = Float64[0, 3, 5]



