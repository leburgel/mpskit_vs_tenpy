# Sub-algorithm settings for the benchmark. One place to change; all scripts use these.
# Rationale and measurements: FINDINGS.md section 3.

module Eigsolvers

using MPSKit, KrylovKit
using MPSKit: Defaults, DynamicTol

export bench_eigsolve, bench_gauge, bench_environments

"""
    bench_eigsolve(; kwargs...)

Local eigensolver: `krylovdim = 16`, `maxiter = 5`, `eager`, single-pass-capable
`orth`, `tol_factor = 1e0`, `tol_max = 1e-2`. 2.5x faster than stock with a lower
energy and a 6x smaller Galerkin error.

`tol_max` is the binding knob: the dynamic tolerance is
`clamp(tol_factor * g_global / sqrt(iter), tol_min, tol_max)`, and with the
Galerkin error at ~1e-3 any `tol_factor >= 1e-1` saturates against the ceiling.

Built by hand rather than via `Defaults.alg_eigsolve` only because that exposes no
`orth`. With these defaults it is bit-identical to
`Defaults.alg_eigsolve(; krylovdim = 16, maxiter = 5, eager = true,
ishermitian = true, tol_factor = 1e0, tol_max = 1e-2)`; `ishermitian` is implicit
in choosing `Lanczos`. Pass `dynamic_tols = false` for the `changebonds`
algorithms, which use static tolerances upstream.
"""
function bench_eigsolve(;
        krylovdim::Int = 16,
        maxiter::Int = 5,
        eager::Bool = true,
        orth = KrylovKit.ModifiedGramSchmidt2(),
        tol::Real = Defaults.tol,
        verbosity::Int = 0,
        dynamic_tols::Bool = true,
        tol_min::Real = Defaults.tol_min,
        tol_max::Real = 1.0e-2,
        tol_factor::Real = 1.0e0,
    )
    alg = Lanczos(; tol, maxiter, eager, krylovdim, orth, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

"""
    bench_environments(; kwargs...)

Environment solver, relaxed like [`bench_eigsolve`]: `krylovdim = 16`,
`maxiter = 5`, `eager`, `tol_factor = 1e0`, `tol_max = 1e-2`. Worth 1.52x on VUMPS.

A separate knob because it governs VUMPS's dominant cost: `envs` is ~50% of an
iteration and runs through `alg_environments`, which carries its own `DynamicTol`.
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

Stock `Defaults.alg_gauge()`. A named pass-through so every solver construction
reads alike and there is one place to change. Gauging is not worth tuning: <=1.3%
of a VUMPS iteration, and `IDMRG2` gauges only once, outside the iteration loop.
"""
bench_gauge(; kwargs...) = Defaults.alg_gauge(; kwargs...)

end # module
