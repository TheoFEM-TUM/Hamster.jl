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

Adaptive interpolation of a given function `f` using an `AdaptiveInterpolator`. The function is sampled 
at adaptive points and interpolated using cubic splines until convergence is reached or the maximum number 
of samples (`Nmax`) is exceeded.

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
    xmin = itper.xmin; xmax = itper.xmax
    Nmax = itper.Nmax
    xs_pool = collect(LinRange(xmin, xmax, Nmax))
    step = floor(Int64, Nmax/itper.Ninit)
    xs = xs_pool[1:step:Nmax]; push!(xs, xmax)
    filter!(x->x∉xs, xs_pool)
    
    ys = f.(xs)

    fint_old = CubicSpline(xs, ys)
    fint_new = CubicSpline(xs, ys)

    diffs = Float64[]; converged = false
    while length(xs) < Nmax && !converged
        ymax = maximum(abs.(ys))
        xnew = get_new_point!(xs_pool, fint_new, xmax, ymax)
        ynew = f(xnew)
        insertxy!(xs, ys, xnew, ynew)
        
        fint_old = fint_new
        fint_new = CubicSpline(xs, ys)
        
        xtrial = LinRange(xmin, xmax, Nmax)
        diff = mean(abs.(fint_old.(xtrial) .- fint_new.(xtrial)))
        push!(diffs, diff)
        if length(diffs) > 5 && all(diffs[end-5:end] .< itper.tol)
            converged=true
        end
    end
    if !converged; @warn "Interpolation did not converge! Maximum number of iterations reached."; end
    return xs, ys
end

"""
    insertxy!(xs, ys, xnew, ynew)

Insert a new point `(xnew, ynew)` into two vectors `xs` and `ys`, which are assumed to be paired coordinates, 
while maintaining the sorted order of `xs`.

# Arguments
- `xs`: A vector of `x` values (assumed to be sorted in ascending order).
- `ys`: A vector of `y` values corresponding to `xs`.
- `xnew`: The new `x` value to insert into `xs`.
- `ynew`: The new `y` value to insert into `ys`, corresponding to `xnew`.
"""
function insertxy!(xs, ys, xnew, ynew)
    if length(xs) == 0; push!(xs, xnew); push!(ys, ynew)
    else 
        for i in eachindex(xs)
            if xs[i] > xnew || i == length(xs); insert!(xs, i, xnew); insert!(ys, i, ynew); break; end
        end
    end
end

"""
    get_new_point(f, xmax, ymax, Nmax; max_iter=10000)

Generate a new point `xnew` for function sampling based on the gradient and curvature of the function `f`. 
This function prioritizes areas of the function with high curvature or gradient, to improve sampling in regions 
with rapid changes.

# Arguments
- `f`: The function to sample from. Assumed to have fields `xs` containing previous sample points.
- `xmax`: The maximum possible value for `x`.
- `ymax`: The maximum possible value of the function `f` over the sampled points.
- `Nmax`: The maximum number of points that can be generated.
- `max_iter`: Maximum number of iterations to attempt generating a new point (default is `10000`).

# Returns
- `xnew`: A new sampled point `xnew` that is in a region with either high gradient or high curvature, and 
  sufficiently far from existing points in `f.xs`.
"""
function get_new_point!(xs_pool, f, xmax, ymax; max_iter=10000)
    grad_max = maximum(abs.([gradient(f, x, 1) for x in f.xs]))
    curv_max = maximum(abs.([gradient(f, x, 2) for x in f.xs]))
    for _ in 1:max_iter, xnew in xs_pool[rand(1:length(xs_pool), length(xs_pool))]
        ynew = abs(f(xnew))
        grad_new = abs(gradient(f, xnew, 1))
        curv_new = abs(gradient(f, xnew, 2))
        p_y = ymax * rand()
        p_grad = grad_max * rand()
        p_curv = curv_max * rand()

        is_large = p_y ≤ ynew
        has_large_gradient = p_grad ≤ grad_new
        has_large_curvature = p_curv ≤ curv_new
        if is_large || has_large_gradient || has_large_curvature
            filter!(x->x≠xnew, xs_pool)
            return xnew
        end
    end
    println("Point generation failed!")
    xnew = xs_pool[rand(1:length(xs_pool))]
    filter!(x->x≠xnew, xs_pool)
    return xnew
end