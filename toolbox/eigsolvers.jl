# Sub-algorithm settings for the benchmark: one place to change, all scripts use
# these. Measurements and rationale in FINDINGS.md section 3.

module Eigsolvers

using MPSKit, KrylovKit
using MPSKit: Defaults, DynamicTol

export bench_eigsolve, bench_gauge, bench_environments

"""
    bench_eigsolve(; kwargs...)

Local eigensolver: `krylovdim = 16`, `maxiter = 5`, `eager`, `tol_factor = 1e0`,
`tol_max = 1e-2`. 1.8x faster than stock, with a lower energy and a 6x smaller
Galerkin error.

`tol_max` is the binding knob: the dynamic tolerance is
`clamp(tol_factor * g_global / sqrt(iter), tol_min, tol_max)`, so with the Galerkin
error at ~1e-3 any `tol_factor >= 1e-1` just saturates against the ceiling.

Equivalent to `Defaults.alg_eigsolve(; krylovdim = 16, maxiter = 5, eager = true,
ishermitian = true, tol_factor = 1e0, tol_max = 1e-2)`, built by hand only so the
`DynamicTol` wrapper is explicit. Pass `dynamic_tols = false` for the `changebonds`
algorithms, which use static tolerances upstream.
"""
function bench_eigsolve(;
        krylovdim::Int = 16,
        maxiter::Int = 5,
        eager::Bool = true,
        tol::Real = Defaults.tol,
        verbosity::Int = 0,
        dynamic_tols::Bool = true,
        tol_min::Real = Defaults.tol_min,
        tol_max::Real = 1.0e-2,
        tol_factor::Real = 1.0e0,
    )
    alg = Lanczos(; tol, maxiter, eager, krylovdim, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

"""
    bench_environments(; kwargs...)

Environment solver, relaxed exactly like [`bench_eigsolve`]. Worth 1.52x on VUMPS,
whose dominant cost it is: `envs` is ~50% of an iteration and runs through
`alg_environments`, which carries its own `DynamicTol`.

Only `VUMPS` exposes the field — `IDMRG`/`IDMRG2` transfer environments
incrementally, and `VUMPSSvdCut` calls bare `environments` internally.
"""
function bench_environments(; kwargs...)
    return Defaults.alg_environments(;
        krylovdim = 16, maxiter = 5, eager = true,
        tol_factor = 1.0e0, tol_max = 1.0e-2, kwargs...
    )
end

"""
    bench_gauge(; kwargs...)

Stock `Defaults.alg_gauge()`, as a named pass-through so every solver construction
reads alike. Gauging is not worth tuning: <=1.3% of a VUMPS iteration, and
`IDMRG2` gauges once, outside the iteration loop.
"""
bench_gauge(; kwargs...) = Defaults.alg_gauge(; kwargs...)

end # module
