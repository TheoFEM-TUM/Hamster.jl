const ke = 14.3996454784255

@inline _wrap01(x) = x - floor(x)
@inline _sinc(x) = abs(x) < 1e-14 ? 1.0 : sin(x)/x

# cubic B-spline weights
# returns weights for offsets (-1, 0, 1, 2)
@inline function _bspline4_weights(t)
    w0 = (1 - t)^3 / 6
    w1 = (3t^3 - 6t^2 + 4) / 6
    w2 = (-3t^3 + 3t^2 + 3t + 1) / 6
    w3 = t^3 / 6
    return (w0, w1, w2, w3)
end

# minimum-image displacement for a general 3x3 lattice box
@inline function _disp_pbc_general(ri::AbstractVector, rj::AbstractVector, box::AbstractMatrix, box_inv::AbstractMatrix)
    dr = ri - rj
    s = box_inv * dr
    s .-= round.(s)
    return box * s
end

function next_fft_size(n)
    # simple version: round up to multiple of 2
    while !issmooth(n)
        n += 1
    end
    return n
end

function issmooth(n)
    for p in (2,3,5)
        while n % p == 0
            n ÷= p
        end
    end
    return n == 1
end

function mesh_from_spacing(box, h)
    L = (norm(box[:,1]), norm(box[:,2]), norm(box[:,3]))

    N = ntuple(i -> ceil(Int, L[i] / h), 3)
    N = ntuple(i -> next_fft_size(N[i]), 3)

    return N
end

# --------------------------------
# Charge spreading with cubic B-splines
# --------------------------------
function _spread_bspline4!(
    rho,
    pos,
    q,
    box,
    box_inv
)
    Nx, Ny, Nz = size(rho)
    fill!(rho, 0.0)

    cellvol = abs(det(box))
    voxelvol = cellvol / (Nx * Ny * Nz)

    N = length(pos)

    @views for i in 1:N
        r = pos[i]
        qi = q[i]

        s = box_inv * r
        sx = _wrap01(s[1]) * Nx
        sy = _wrap01(s[2]) * Ny
        sz = _wrap01(s[3]) * Nz

        ix = floor(Int, sx)
        iy = floor(Int, sy)
        iz = floor(Int, sz)

        tx = sx - ix
        ty = sy - iy
        tz = sz - iz

        wx = _bspline4_weights(tx)
        wy = _bspline4_weights(ty)
        wz = _bspline4_weights(tz)

        for (ax, ox) in enumerate((-1, 0, 1, 2))
            gx = mod(ix + ox, Nx) + 1
            for (ay, oy) in enumerate((-1, 0, 1, 2))
                gy = mod(iy + oy, Ny) + 1
                for (az, oz) in enumerate((-1, 0, 1, 2))
                    gz = mod(iz + oz, Nz) + 1
                    rho[gx, gy, gz] += qi * wx[ax] * wy[ay] * wz[az]
                end
            end
        end
    end

    rho ./= voxelvol
    return rho
end

# --------------------------------
# Gather potential back to particles
# --------------------------------
function _gather_bspline4(
    field,
    pos,
    box_inv
)
    Nx, Ny, Nz = size(field)
    N = length(pos)
    vals = zeros(eltype(field), N)

    @views for i in 1:N
        r = pos[i]
        s = box_inv * r

        sx = _wrap01(s[1]) * Nx
        sy = _wrap01(s[2]) * Ny
        sz = _wrap01(s[3]) * Nz

        ix = floor(Int, sx)
        iy = floor(Int, sy)
        iz = floor(Int, sz)

        tx = sx - ix
        ty = sy - iy
        tz = sz - iz

        wx = _bspline4_weights(tx)
        wy = _bspline4_weights(ty)
        wz = _bspline4_weights(tz)

        acc = 0.0
        for (ax, ox) in enumerate((-1, 0, 1, 2))
            gx = mod(ix + ox, Nx) + 1
            for (ay, oy) in enumerate((-1, 0, 1, 2))
                gy = mod(iy + oy, Ny) + 1
                for (az, oz) in enumerate((-1, 0, 1, 2))
                    gz = mod(iz + oz, Nz) + 1
                    acc += wx[ax] * wy[ay] * wz[az] * field[gx, gy, gz]
                end
            end
        end

        vals[i] = acc
    end

    return vals
end