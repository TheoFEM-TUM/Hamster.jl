function distance_dependence(ME::MatrixElement, orbconfig::OrbitalConfiguration, r, ns, αs)
    R₁ = ndict[ns[1]](αs[1]); R₂ = ndict[ns[2]](αs[2])
    xmin = [-10, -10, -10]; xmax = [15+r/3, 15+r/3, r+15]
    param = string(ME)
    m = mdict_inv[param[3]]
    l₁, l₂ = typeof(orbconfig) == MirrOrb ? (ldict_inv[param[2]], ldict_inv[param[1]]) : (ldict_inv[param[1]], ldict_inv[param[2]])
    sign = l₁ > l₂ ? (-1)^(l₁+l₂) : 1
    Y₁, Y₂ = get_spherical.([l₁, l₂], m)
    f = Overlap(Y₁, R₁, zeros(3), Y₂, R₂, [0., 0., r])
    I, _ = hcubature(f, xmin, xmax, rtol=1e-5, maxevals=1000000, initdiv=5)
    return sign*I
end