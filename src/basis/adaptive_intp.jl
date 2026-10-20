"""
    AdaptiveInterpolator(xmin::Float64, xmax::Float64, Ninit::Int64, Nmax::Int64, tol::Float64)

A struct that defines the settings for adaptive interpolation.

# Fields
- `xmin::Float64`: The minimum value of `x` in the interpolation range.
- `xmax::Float64`: The maximum value of `x` in the interpolation range.
- `Ninit::Int64`: The initial number of sample points for the interpolation.
- `Nmax::Int64`: The maximum number of sample points allowed during the interpolation process.
- `tol::Float64`: The convergence tolerance. If the interpolation error falls below this value, the algorithm is considered to have converged.
"""
struct AdaptiveInterpolator
    xmin :: Float64
    xmax :: Float64
    Ninit :: Int64
    Nmax :: Int64
    tol :: Float64
end

AdaptiveInterpolator(conf=get_empty_config(); xmin=get_itp_xmin(conf), 
    xmax=get_itp_xmax(conf), Nmax=get_itp_Nmax(conf), Ninit=get_itp_Ninit(conf), tol=get_itp_tol(conf)) = AdaptiveInterpolator(xmin, xmax, Ninit, Nmax, tol)

"""
    interpolate_f(itper::AdaptiveInterpolator, f::Function)

Adaptive interpolation of a given function `f` with a cubic spline by bisection: starting from `Ninit` equidistant
intervals, `f` is evaluated at the midpoints of all intervals that are not yet converged and compared with the spline
of the current points. Every midpoint becomes a new point; an interval whose midpoint is predicted within `tol` is
converged (both halves), the others are refined further, until all intervals are converged or `Nmax` points are
reached. So `tol` bounds the interpolation error (checked at the midpoints, which are then added), the points
concentrate where `f` is hard to interpolate, and the spline is rebuilt only once per round. The result is
deterministic.

# Arguments
- `itper::AdaptiveInterpolator`: An instance of the `AdaptiveInterpolator` struct that defines the interpolation 
  settings, such as the initial number of points, the maximum allowed number of points (`Nmax`), 
  and the convergence tolerance.
- `f::Function`: The function to interpolate.

# Returns
- `xs`: A vector of `x` values where the function has been evaluated.
- `ys`: A vector of `y` values (`f(xs)`) corresponding to the `x` values.
"""
function interpolate_f(itper::AdaptiveInterpolator, f::Function)
    xs = collect(range(itper.xmin, itper.xmax, length=itper.Ninit + 1))
    ys = f.(xs)
    active = trues(length(xs) - 1)              # active[i]: interval (xs[i], xs[i+1]) is not converged
    while any(active) && length(xs) < itper.Nmax
        spline = CubicSpline(xs, ys)
        todo = findall(active)
        mids = [(xs[i] + xs[i+1]) / 2 for i in todo]
        ymids = f.(mids)
        converged = abs.(ymids .- spline.(mids)) .< itper.tol
        # Insert the midpoints from the right (the indices of the intervals to the left stay valid)
        for n in reverse(eachindex(todo))
            i = todo[n]
            insert!(xs, i + 1, mids[n]); insert!(ys, i + 1, ymids[n])
            active[i] = !converged[n]; insert!(active, i + 1, !converged[n])
        end
    end
    if any(active); @warn "Interpolation did not converge! Maximum number of points reached."; end
    return xs, ys
end
