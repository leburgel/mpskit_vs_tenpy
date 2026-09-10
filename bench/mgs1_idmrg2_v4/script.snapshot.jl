using JLD2, LinearAlgebra
using Dates, TOML
using Plots
using Logging, Printf, NPZ

include(joinpath(@__DIR__, "toolbox", "Toolbox.jl")); using .Toolbox

# Single core (FINDINGS.md section 10.1: threads buy nothing here).
# BENCH_BLAS_CAP raises it if ever needed.
cap_blas_threads()

## --- Parameters for Haldane model --

t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1
Nx = 1
Ny = 6 # circumference of cylinder (number of unit cells in y direction)
## --- build FCI Hamiltonian ---
H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)

# start from CDW product state, expanded through H, then perturbed
psi1 = initialstate_product(H_FCI_shifted, Nx, Ny)

# DMRG warmup
chi_warmup = 100
tol_warmup = 1.0e-4
maxiter_warmup = 50
# Verbosity is a knob because it decides whether the solver is instrumented at
# all, so the same file serves both kinds of run:
#   BENCH_VERBOSITY=4 (default) -- live TimerOutput, per-stage split, some overhead
#   BENCH_VERBOSITY=3           -- NoTimerOutput, zero timer overhead, clean totals
# Use 3 for any absolute number quoted against TeNPy, 4 for the stage breakdown.
verbosity_warmup = parse(Int, get(ENV, "BENCH_VERBOSITY", "4"))
# 4, not 3, deliberately: MPSKit only *creates* a live TimerOutput when
# `verbosity > 3` (`timeroutput = alg.verbosity > 3 ? TimerOutput(...) :
# NoTimerOutput()`, vumps.jl:67 / idmrg.jl:131). At verbosity 3 every `@timeit`
# in the solver is a no-op on a NoTimerOutput -- so there is no per-stage
# breakdown to report, and equally no timer overhead in the runs that used it.
# At 4 the solver prints a `TimerReport` per `find_groundstate` call (via @infov,
# i.e. to stderr, so it lands in time.log) splitting localupdate / gauge / envs /
# finalize / calc_galerkin with call counts.

# Orthogonalizer under test, selected by BENCH_ORTH so no source edit is needed
# between runs and the choice lands in the log:
#   mgs2 (default) -- ModifiedGramSchmidt2, two-pass, MPSKit's stock choice
#   mgs            -- ModifiedGramSchmidt, single-pass, the cheapest KrylovKit has
# Everything else about the eigensolver is identical between the two.
const ORTH = get(ENV, "BENCH_ORTH", "mgs2")
eigsolve_alg = if ORTH == "mgs"
    bench_eigsolve(; orth = KrylovKit.ModifiedGramSchmidt())
elseif ORTH == "mgs2"
    bench_eigsolve()
else
    error("BENCH_ORTH must be \"mgs\" or \"mgs2\", got \"$ORTH\"")
end
println("eigensolver: ", eigsolve_alg)

tracker = EnergyTracker(; nsites = Nx * Ny * 2)

alg = IDMRG2(;
    trunc = truncrank(chi_warmup),
    maxiter = maxiter_warmup,
    tol = tol_warmup,
    verbosity = verbosity_warmup,
    alg_eigsolve = eigsolve_alg,
    alg_gauge = bench_gauge(),
    finalize = tracker
)

t_solve = @elapsed psi1, envs, err = find_groundstate(psi1, H_FCI_shifted, alg)
times, energies = tracker.times, tracker.energies

# The tracked per-iteration energies are IDMRG2's own ΔE-per-unit-cell estimator
# (what upstream logs as `IterLog.objective`), which is what TeNPy's sweep_stats
# also reports -- but it is not a variational expectation value, and at low chi it
# can sit *below* the true ground-state energy (measured: -0.28294 at chi = 4,
# lower than TeNPy's converged -0.27923). So print the variational number too.
# `envs` is the one find_groundstate just returned, so it matches psi1 and this
# costs no environment rebuild.
E_var = real(expectation_value(psi1, H_FCI_shifted, envs)) / (Nx * Ny * 2)
@printf("\nfind_groundstate: %.2f s, %d iterations, final Galerkin error %.3e\n",
        t_solve, length(times), err)
@printf("  E/site, IDMRG2 dE estimator = %+.10f\n", isempty(energies) ? NaN : energies[end])
@printf("  E/site, variational         = %+.10f\n", E_var)
println("  reference: TeNPy -0.2792287344 (converged, 22 sweeps)")
println("             IDMRG2 5 iterations -0.2765905800 (not converged)\n")

println(rpad("iter", 6), rpad("time [s]", 12), "energy")
println("-"^40)
for (i, (t, E)) in enumerate(zip(times, energies))
    @printf("%-6d%-12.3f%.8f\n", i, t, E)
end

p = plot(
    times, energies; xlabel = "elapsed wall time (s)", ylabel = "Energy",
    marker = :circle, legend = false, title = "IDMRG2 warmup convergence"
)
savefig(p, joinpath(@__DIR__, "energy_vs_time.png"))

npzwrite(joinpath(@__DIR__, "energy_vs_time.npy"), hcat(times, energies))
