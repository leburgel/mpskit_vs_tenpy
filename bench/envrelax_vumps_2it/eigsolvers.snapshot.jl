# Local eigensolver configurations, for isolating the one eigensolver difference
# that survives at every bond dimension.
#
# FINDINGS.md section 5.4: MPSKit and TeNPy both run Lanczos on this benchmark at
# chi=100 (TeNPy's dense-ED branch needs 4*chi^2 < 400, i.e. chi <= 9), but they
# configure it very differently:
#
#            | MPSKit                  | TeNPy
#   krylovdim| 30                      | 20  (N_max)
#   reortho  | ModifiedGramSchmidt2    | none (reortho = False)
#   restarts | maxiter = 200           | none
#   eager    | true                    | -
#
# `ModifiedGramSchmidt2` reorthogonalizes against the whole Krylov basis with a
# second pass always taken; `ModifiedGramSchmidt` is a single pass and is the
# cheapest orthogonalizer KrylovKit offers -- there is no "no reorthogonalization"
# option, since Lanczos needs at least the local orthogonalization to build the
# tridiagonal form. So `tenpy_like` is the closest available match, not an exact one.

module Eigsolvers

using MPSKit, KrylovKit
using MPSKit: Defaults, DynamicTol

export lanczos_eigsolve, mpskit_default_eigsolve, tenpy_like_eigsolve
export bench_eigsolve, bench_gauge, bench_environments

"""
    lanczos_eigsolve(; krylovdim, orth, eager, tol, maxiter, dynamic_tols = true)

A Lanczos local eigensolver, wrapped in `DynamicTol` exactly as
`Defaults.alg_eigsolve` does, so the outer iteration still adapts its tolerance.

Exists because `Defaults.alg_eigsolve` takes no `orth` argument, which is the
parameter under test here.
"""
function lanczos_eigsolve(;
        krylovdim::Int = Defaults.krylovdim,
        orth = KrylovKit.ModifiedGramSchmidt2(),
        eager::Bool = true,
        tol::Real = Defaults.tol,
        maxiter::Int = Defaults.maxiter,
        verbosity::Int = 0,
        dynamic_tols::Bool = true,
    )
    alg = Lanczos(; tol, maxiter, eager, krylovdim, orth, verbosity)
    return dynamic_tols ?
        DynamicTol(alg, Defaults.tol_min, Defaults.tol_max, Defaults.eigs_tolfactor) : alg
end

"""MPSKit's stock configuration: krylovdim 30, `ModifiedGramSchmidt2`, 200 restarts."""
mpskit_default_eigsolve(; kwargs...) = lanczos_eigsolve(;
    krylovdim = 30, orth = KrylovKit.ModifiedGramSchmidt2(), maxiter = 200, kwargs...
)

"""
TeNPy's configuration, as closely as KrylovKit allows: Krylov dimension 20, the
cheapest single-pass orthogonalizer, and no restarts (`maxiter = 1`).
"""
tenpy_like_eigsolve(; kwargs...) = lanczos_eigsolve(;
    krylovdim = 20, orth = KrylovKit.ModifiedGramSchmidt(), maxiter = 1, kwargs...
)

# The configuration the benchmark runs, set 2026-09-08.
#
# These go through `Defaults.alg_eigsolve` / `Defaults.alg_gauge` rather than
# `lanczos_eigsolve` above, because `tol_factor` is a property of the `DynamicTol`
# wrapper those constructors apply, not of the bare `Lanczos`:
#
#     tol = clamp(tol_factor * g_global / sqrt(iter), tol_min, tol_max)
#
# so `tol_factor` scales how aggressively the local tolerance tracks the global
# convergence error (dynamictols.jl:87). Lowering the gauge factor to 1e-3 gauges
# more tightly than the solver converges.
#
# `tol_max` is the binding constraint, not `tol_factor`. Stock values are
# `tol_factor = 1e-3`, `tol_min = 1e-14`, `tol_max = 1e-4` (defaults.jl:28-30), so
# with the Galerkin error at ~8e-3 the stock local tolerance is
# 1e-3 * 8e-3 / sqrt(50) ~ 1e-6. Raising the factor to 1e-1 puts the product at
# ~1.1e-4, i.e. *above* `tol_max`, so the tolerance saturates at 1e-4 -- a ~100x
# loosening, which is where the 1.68x speedup to 8.12 s/iteration came from
# (FINDINGS 12.10). Raising the factor further to 1e0 changes nothing while the
# clamp holds. Hence `tol_max = 1e-2` here: it lifts the ceiling by another 100x
# and is the only way to go looser still. Watch the Galerkin error and the final
# energy, not just s/iteration -- this is the setting that can make each local
# solve too inaccurate for the outer iteration to make progress.

"""
    bench_eigsolve(; kwargs...)

The benchmark's local eigensolver: `krylovdim = 16`, `maxiter = 5`, `eager`,
Hermitian, `tol_factor = 1e0`, `tol_max = 1e-2`. Any keyword overrides the corresponding default —
pass `dynamic_tols = false` for the `changebonds` algorithms, which are built with
static tolerances upstream (and for which `tol_factor` is therefore inert).
"""
bench_eigsolve(; kwargs...) = Defaults.alg_eigsolve(;
    krylovdim = 16, maxiter = 5, eager = true, ishermitian = true,
    tol_factor = 1.0e0, tol_max = 1.0e-2, kwargs...
)

"""
    bench_gauge(; kwargs...)

The benchmark's gauging algorithm: **stock**, i.e. `Defaults.alg_gauge()`.

Kept as a named pass-through so every solver construction in the benchmark reads
the same way and there is one place to change if gauging ever needs tuning. It
briefly carried `tol_factor = 1e-3`; that was dropped on 2026-09-08 because
gauging is not a cost worth tuning:

  * VUMPS spends **1.3%** of an iteration in `gauge` (7.05 s of 541 s; its
    `gauge_orth` 1.1% + `gauge_eigsolve` 0.2%) — FINDINGS 12.7.
  * IDMRG2 does not gauge inside the iteration at all. `alg_gauge` is used only
    in the final `InfiniteMPS(state.mps.AR; alg_gauge.tol, alg_gauge.maxiter)`
    (`idmrg.jl:165`), outside the loop, which is why `gauge` never appears in its
    TimerReport.

The caveat, untested: gauge tolerance is not purely a cost knob — gauge quality
feeds the environment solves and the Galerkin error, so it could matter for
*stability* (the VUMPS-on-OptimalExpand divergence of 12.9 is the obvious place
it might). Nothing measured showed such an effect either way.
"""
bench_gauge(; kwargs...) = Defaults.alg_gauge(; kwargs...)

"""
    bench_environments(; kwargs...)

The benchmark's environment solver, relaxed the same way as [`bench_eigsolve`]:
`krylovdim = 16`, `maxiter = 5`, `eager`, `tol_factor = 1e0`, `tol_max = 1e-2`.
(No `ishermitian` — `Defaults.alg_environments` builds a `DefaultAlgorithm`, not a
`Lanczos`/`Arnoldi` pair, so it takes no such keyword.)

Why this exists as a separate knob: **`alg_eigsolve` does not govern VUMPS's
dominant cost.** Per FINDINGS 12.7, a VUMPS iteration is 50.6% `envs` against
45.7% `localupdate`, and the `envs` stage runs `recalculate!` through
`alg_environments`, which carries its *own* `DynamicTol` with its own
`tol_factor = envs_tolfactor` (`defaults.jl:78-84`). That is why relaxing
`alg_eigsolve` bought IDMRG2 1.85x (15.0 -> 8.1 s/iteration) but VUMPS almost
nothing (10.8 -> 10.2): it was tuning the smaller half of the wrong profile.

Only `VUMPS` exposes this field. `IDMRG`/`IDMRG2` have no `alg_environments` (they
update environments incrementally, 3.7% of an iteration, so there is nothing to
tune), and `VUMPSSvdCut` calls bare `environments(state, H, state)` inside
`changebonds_n`, so its rebuilds — the ~210 s/call of section 11.3 — are *not*
reachable through this and remain untunable without patching MPSKit.
"""
bench_environments(; kwargs...) = Defaults.alg_environments(;
    krylovdim = 16, maxiter = 5, eager = true,
    tol_factor = 1.0e0, tol_max = 1.0e-2, kwargs...
)

end # module
