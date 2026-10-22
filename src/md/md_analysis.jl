"""
    md_rdf(positions, lattice, types; rmax=8.0, nbins=400)

Partial radial distribution functions g_AB(r) of the frames `positions` (3 × N × Nf) for all species pairs (A ≤ B) and
the total one. Returns `(r, g::Dict{String, Vector{Float64}})` with keys like "Pb-Br" and "total".
"""
function md_rdf(positions, lattice, types; rmax=8.0, nbins=400)
    L = SMatrix{3, 3, Float64}(lattice)
    V = abs(det(L))
    species = sort(unique(types))
    pairs = [(a, b) for (i, a) in enumerate(species) for b in species[i:end]]
    counts = Dict(P => zeros(nbins) for P in pairs)
    total = zeros(nbins)
    dr = rmax / nbins
    Nf = size(positions, 3)
    for n in 1:Nf
        x = positions[:, :, n]
        nl = NeighborList(x, L, rmax; rmin=1e-8)
        for p in eachindex(nl.i)
            r = norm(bond_vector(x, L, nl.i[p], nl.j[p], nl.n[p]))
            k = floor(Int, r / dr) + 1
            k ≤ nbins || continue
            a, b = types[nl.i[p]], types[nl.j[p]]
            counts[a ≤ b ? (a, b) : (b, a)][k] += 1
            total[k] += 1
        end
    end
    r = [(k - 0.5) * dr for k in 1:nbins]
    shell = [4π / 3 * ((k * dr)^3 - ((k - 1) * dr)^3) for k in 1:nbins]
    g = Dict{String, Vector{Float64}}()
    Ns = Dict(t => count(==(t), types) for t in species)
    for (a, b) in pairs
        # directed pairs: (i ∈ A, j ∈ B) and (i ∈ B, j ∈ A) are both counted for A ≠ B
        norm_ab = (a == b ? Ns[a]^2 : 2Ns[a] * Ns[b]) / V
        g["$(number_to_element(a))-$(number_to_element(b))"] = counts[(a, b)] ./ (Nf * norm_ab .* shell)
    end
    g["total"] = total ./ (Nf * length(types)^2 / V .* shell)
    return r, g
end

"""
    md_vdos(velocities, masses, types, dt; window=true)

Vibrational density of states from the velocities (3 × N × Nt, Å/fs, frames `dt` fs apart): the mass-weighted power
spectrum of the velocities (Fourier transform of the velocity autocorrelation), per species and in total, normalized
to 3 per atom. Returns `(ν_THz, vdos::Dict{String, Vector{Float64}})`, and the normalized velocity autocorrelation.
"""
function md_vdos(velocities, masses, types, dt; window=true)
    N, Nt = size(velocities, 2), size(velocities, 3)
    Npad = 2Nt
    w = window ? [0.5 - 0.5cos(2π * (t - 1) / (Nt - 1)) for t in 1:Nt] : ones(Nt)
    ν = collect(0:Nt) ./ (Npad * dt) .* 1000          # 1/fs → THz
    spectra = Dict{UInt8, Vector{Float64}}()
    vacf = zeros(Nt)
    buf = zeros(ComplexF64, Npad)
    for a in 1:N, x in 1:3
        fill!(buf, 0)
        buf[1:Nt] .= velocities[x, a, :] .* w
        f = fft(buf)
        p = masses[a] .* abs2.(f[1:Nt+1])
        spectra[types[a]] = get(spectra, types[a], zeros(Nt + 1)) .+ p
        ac = real(ifft(abs2.(fft(vcat(velocities[x, a, :], zeros(Nt))))))[1:Nt]
        vacf .+= masses[a] .* ac
    end
    dν = ν[2] - ν[1]
    out = Dict{String, Vector{Float64}}()
    tot = zeros(Nt + 1)
    for (t, s) in spectra
        nat = count(==(t), types)
        sn = s ./ (sum(s) * dν) .* 3nat
        out[string(number_to_element(t))] = sn
        tot .+= sn
    end
    out["total"] = tot
    vacf ./= vacf[1]
    return ν, out, vacf
end

"""
    md_msd(positions, types)

Mean squared displacement (Å²) of every species with respect to the first frame (unwrapped positions).
"""
function md_msd(positions, types)
    Nf = size(positions, 3)
    out = Dict{String, Vector{Float64}}()
    for t in unique(types)
        atoms = findall(==(t), types)
        out[string(number_to_element(t))] = [mean(sum(abs2, positions[:, a, n] - positions[:, a, 1]) for a in atoms) for n in 1:Nf]
    end
    return out
end

"""
    md_analysis(filename; equilibration=0, rmax=8.0, nbins=400)

Analysis of an MD output file: mean and standard deviation of the temperature and pressure, radial distribution
functions, vibrational density of states and mean squared displacements of the frames after `equilibration` steps.
"""
function md_analysis(filename; equilibration=0, rmax=8.0, nbins=400)
    h5open(filename, "r") do f
        steps = read(f["thermo/step"])
        sel = steps .≥ equilibration
        T = read(f["thermo/temperature"])[sel]; P = read(f["thermo/pressure"])[sel]
        types = UInt8.(read(f["types"])); masses = read(f["masses"]); lattice = read(f["lattice"])
        tsteps = read(f["trajectory/step"])
        tsel = findall(≥(equilibration), tsteps)
        x = read(f["trajectory/positions"])[:, :, tsel]
        v = read(f["trajectory/velocities"])[:, :, tsel]
        times = read(f["trajectory/time"])[tsel]
        r, g = md_rdf(x, lattice, types; rmax=rmax, nbins=nbins)
        ν, vdos, vacf = length(tsel) > 2 ? md_vdos(v, masses, types, times[2] - times[1]) : (Float64[], Dict{String, Vector{Float64}}(), Float64[])
        msd = md_msd(x, types)
        return (T_mean=mean(T), T_std=std(T), P_mean=mean(P), P_std=std(P), r=r, rdf=g, nu=ν, vdos=vdos, vacf=vacf,
                time=times .- times[1], msd=msd)
    end
end

"""
    write_md_analysis(filename, analysis)

Write the result of `md_analysis` to the group `analysis` of the MD output file.
"""
function write_md_analysis(filename, a)
    h5open(filename, "r+") do f
        haskey(f, "analysis") && delete_object(f, "analysis")
        g = create_group(f, "analysis")
        g["T_mean"] = a.T_mean; g["T_std"] = a.T_std; g["P_mean"] = a.P_mean; g["P_std"] = a.P_std
        g["rdf/r"] = a.r
        for (k, v) in a.rdf; g["rdf/$k"] = v; end
        g["vdos/nu_THz"] = a.nu
        for (k, v) in a.vdos; g["vdos/$k"] = v; end
        g["vacf"] = a.vacf; g["time"] = a.time
        for (k, v) in a.msd; g["msd/$k"] = v; end
    end
end
