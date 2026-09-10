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
L = Nx * Ny * 2
## --- build FCI Hamiltonian ---
H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)

# start from CDW product state, expanded through H, then perturbed
psi1 = initialstate_product(H_FCI_shifted, Nx, Ny)

# DMRG warmup
chi_warmup = 100
tol_warmup = 1.0e-4
verbosity_warmup = 4
# 4, not 3, deliberately: MPSKit only *creates* a live TimerOutput when
# `verbosity > 3` (`timeroutput = alg.verbosity > 3 ? TimerOutput(...) :
# NoTimerOutput()`, vumps.jl:67 / idmrg.jl:131). At verbosity 3 every `@timeit`
# in the solver is a no-op on a NoTimerOutput -- so there is no per-stage
# breakdown to report, and equally no timer overhead in the runs that used it.
# At 4 the solver prints a `TimerReport` per `find_groundstate` call (via @infov,
# i.e. to stderr, so it lands in time.log) splitting localupdate / gauge / envs /
# finalize / calc_galerkin with call counts.

# Rounds of (VUMPS -> expand). VUMPSSvdCut can at best double a bond per call
# (d = 2 here), so saturating chi 1 -> 100 needs ~7 rounds at best; the extra
# headroom covers bonds that grow more slowly under the symmetry constraints.
# This is only a ceiling: the loop stops as soon as every bond sits at chi_warmup,
# or as soon as a round grows no bond at all.
maxiter_warmup = 50

# VUMPS iterations per round, i.e. between successive changebonds calls.
#
# VUMPSSvdCut picks its new directions from the *current* state, so the more
# optimized the state is when it expands, the better those directions should be.
# Measured both ways (FINDINGS.md 12.8): 2 reached -0.2790950772 against 1's
# -0.2790292232, and converged further (err 2.6e-3 vs 3.3e-3), for ~1.6x the
# warm-up cost. Back to 2 on the strength of that.
vumps_per_round = 2

# Iterations for the final plain-VUMPS convergence run.
vumps_maxiter = 50

tracker = EnergyTracker(; nsites = L)

# One-site VUMPS plus explicit bond expansion, in place of a single IDMRG2 call.
#
# Expansion is VUMPSSvdCut rather than OptimalExpand. OptimalExpand confines the
# new directions to the null space of the current state, which at bond dimension
# chi admits only chi(d-1) of them, so it grows chi by at best a factor of 2 per
# call and in practice much less: measured 1 -> 2 -> 2 -> 2 -> 4 over five rounds,
# reaching only chi = 5 and E/site = -0.2308 after five outer iterations.
#
# VUMPSSvdCut instead builds the full two-site tensor, solves the two-site
# eigenproblem and truncates the SVD to `trunc` -- the same mechanism IDMRG2 uses,
# so a bond can jump to min(chi*d, d*chi). It also sweeps every site within one
# call, updating the state as it goes, so the growth compounds.
#
# It truncates to `trunc` itself, so no separate SvdCut is needed.
alg = VUMPS(; maxiter = vumps_per_round, tol = tol_warmup,
            verbosity = verbosity_warmup, alg_eigsolve = bench_eigsolve(),
            alg_gauge = bench_gauge(), alg_environments = bench_environments(),
            finalize = tracker)
# `changebonds` algorithms are built with `dynamic_tols = false` upstream, so
# `tol_factor` is inert for them; pass it through so the krylovdim/maxiter/eager
# part still applies without silently switching VUMPSSvdCut to dynamic tolerances.
alg_expand = VUMPSSvdCut(;
    trunc = truncrank(chi_warmup),
    alg_eigsolve = bench_eigsolve(; dynamic_tols = false),
    alg_gauge = bench_gauge(; dynamic_tols = false),
)

bond_dims(psi) = [dim(left_virtualspace(psi, i)) for i in 1:length(psi)]

"""
Run up to `maxrounds` rounds of (`vumps_per_round` VUMPS sweeps -> expand).

Written as a function so the loop variables stay local; the tracker is attached to
`alg` as its `finalize`, so it records every VUMPS sweep across all the calls.

Returns the state, its environments, the last Galerkin error, and the per-round
E/site trace. Each round's energy is an `expectation_value` against the
environments VUMPS just returned, so it matches `psi` and costs no rebuild --
unlike the tracker, which records NaN for VUMPS because upstream's 4-argument
`finalize` hands over no energy (FINDINGS.md 11.1).
"""
function warmup_loop(psi, H, alg, alg_expand, chi_max, maxrounds, nsites)
    envs = environments(psi, H, psi)
    err = Inf
    energies = Float64[]
    for round in 1:maxrounds
        psi, envs, err = find_groundstate(psi, H, alg, envs)

        E_site = real(expectation_value(psi, H, envs)) / nsites
        push!(energies, E_site)

        prev = bond_dims(psi)
        @printf(
            "  round %2d: err = %.3e, E/site = %+.10f, min chi = %3d, max chi = %3d, %s\n",
            round, err, E_site, minimum(prev), maximum(prev), string(prev)
        )

        # Every bond at the target: the warm-up is done, stop expanding.
        if minimum(prev) >= chi_max
            println("  (saturated -- every bond at chi = $chi_max)")
            break
        end

        psi, envs = changebonds(psi, H, alg_expand, envs)
        cur = bond_dims(psi)
        @printf("      expanded -> min chi = %3d, max chi = %3d, %s\n",
                minimum(cur), maximum(cur), string(cur))
        if cur == prev
            println("  (stalled -- no bond grew)")
            break
        end
    end
    return psi, envs, err, energies
end

println("\n=== warm-up loop ($(vumps_per_round) VUMPS iterations per expansion) ===")
t_warmup = @elapsed psi1, envs, err, round_energies = warmup_loop(
    psi1, H_FCI_shifted, alg, alg_expand, chi_warmup, maxiter_warmup, L
)
@printf(
    "\nafter expansion loop: %.2f s, err = %.3e, bond dims = %s\n",
    t_warmup, err, string(bond_dims(psi1))
)
@printf(
    "  all bonds at chi = %d: %s\n", chi_warmup,
    minimum(bond_dims(psi1)) >= chi_warmup ? "YES" : "NO"
)

# Proof of concept: once the expansion loop has grown the bonds to chi_warmup,
# hand the state to plain VUMPS and let it actually converge. The question is only
# whether this route reaches the right energy, not whether it is fast.
alg_converge = VUMPS(;
    maxiter = vumps_maxiter, tol = tol_warmup, verbosity = verbosity_warmup,
    alg_eigsolve = bench_eigsolve(), alg_gauge = bench_gauge(),
    alg_environments = bench_environments(), finalize = tracker
)
t_converge = @elapsed psi1, envs, err = find_groundstate(
    psi1, H_FCI_shifted, alg_converge, envs
)
E_final = real(expectation_value(psi1, H_FCI_shifted, envs)) / L
@printf(
    "\nVUMPS(%d): %.2f s, final err = %.3e, E/site = %+.10f\n",
    vumps_maxiter, t_converge, err, E_final
)
@printf("  bond dims = %s\n", string(bond_dims(psi1)))
println("  (TeNPy reference: -0.2792287344;  IDMRG2 product start: -0.2765905800)\n")

times = tracker.times

# The tracker's energies are NaN for VUMPS (upstream's 4-argument finalize hands
# over none), so the per-iteration column here is wall time and bond dimension;
# per-round energies are `round_energies` above and per-iteration ones are in the
# log as `obj = ...`.
@printf("%-6s%-12s%s\n", "iter", "time [s]", "max chi")
println("-"^34)
for k in eachindex(times)
    @printf("%-6d%-12.3f%d\n", k, times[k], maximum(tracker.bonddims[k]))
end

println("\nper-round E/site during the warm-up loop:")
for (r, E) in enumerate(round_energies)
    @printf("  round %2d: %+.10f\n", r, E)
end

p = plot(
    eachindex(round_energies), round_energies;
    xlabel = "warm-up round", ylabel = "E/site",
    marker = :circle, legend = false, title = "VUMPS + VUMPSSvdCut warm-up"
)
savefig(p, joinpath(@__DIR__, "energy_vs_round_me.png"))

npzwrite(
    joinpath(@__DIR__, "me_trace.npz"),
    Dict(
        "times" => times,
        "maxchi" => Float64[maximum(b) for b in tracker.bonddims],
        "minchi" => Float64[minimum(b) for b in tracker.bonddims],
        "round_energies" => round_energies,
        "E_final" => [E_final],
    )
)
println("\nsaved me_trace.npz")
