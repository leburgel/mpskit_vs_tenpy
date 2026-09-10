# Third warm-up route: saturate the bond dimension with repeated OptimalExpand
# steps, then hand the state to plain VUMPS.
#
# Contrast with the other two:
#   script.jl       IDMRG2 from the product state -- two-site updates both grow the
#                   bonds and optimize, reaching chi=100 in ~3 iterations.
#   script_me.jl    VUMPS + VUMPSSvdCut per iteration -- the expansion is a full
#                   two-site update, so chi doubles per iteration, but each call
#                   rebuilds every environment once per site (FINDINGS.md 11.3).
#   script_meme.jl  (this) all the expansion up front, then untouched VUMPS.
#
# Why the noise. `OptimalExpand` is deliberately state-preserving: the added
# directions attach through a *zero* block, so the state is unchanged and the bond
# matrices come out singular. Without a perturbation the next expansion sees the
# same two-site complement and the loop stalls (and VUMPS/IDMRG2 hit
# SingularException). A tiny symmetric perturbation, symmetry- and
# filling-preserving, makes the bonds full rank again -- the same trick
# `product_start.jl` uses once, applied every round here.
#
# Growth rate. One OptimalExpand round can add at most dim(null space) = chi*(d-1)
# directions per bond, i.e. it can at best double chi. Saturating 1 -> 100 therefore
# takes ~7 rounds at best, more with symmetry constraints, which is why `add`
# defaults to chi_max rather than a small number: the null space is the binding
# constraint early on, not `trunc` (measured: truncrank(4), (14), (40) and (100)
# all give an identical chi=2 first round).

using LinearAlgebra
using Plots
using Printf, NPZ

include(joinpath(@__DIR__, "toolbox", "Toolbox.jl")); using .Toolbox

# Single core (FINDINGS.md section 10.1: threads buy nothing here).
cap_blas_threads()

## --- Parameters for Haldane model --
t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1
Nx = 1
Ny = 6
L = Nx * Ny * 2

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

expand_add = chi_warmup     # let the null space, not `trunc`, be the limit
# Perturbation size. Back to 1e-6: raising it to 1e-3 was tested and **refuted**
# the ill-conditioning hypothesis (FINDINGS 12.9). It did improve the expansion
# phase -- per-round energies near-monotonic, -0.2500 after expansion against
# -0.2343 at 1e-6 -- but the VUMPS that follows then *diverged*, -0.2514 at
# iteration 32 to -0.0849 at 50 with err climbing to 6.9e-1, i.e. worse than
# 1e-6's -0.2492. Whatever breaks one-site VUMPS on OptimalExpand-built states,
# perturbation magnitude is not it. Leading remaining guess: the bonds are padded
# from chi=53 to 106 and trimmed to 100, so the state carries nominal chi=100 at
# much lower effective rank, and VUMPS's gauge needs C^-1. Untested.
expand_eps = 1.0e-6
expand_maxrounds = 40
vumps_maxiter = 50
# VUMPS iterations per expansion round (0 reproduces the original
# state-preserving loop, which is what FINDINGS 12.3's -0.2776 came from).
vumps_per_round = 2

## --- build FCI Hamiltonian ---
t_mpo = @elapsed begin
    H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
    H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)
end
@printf("HaldaneMPO build: %.2f s\n", t_mpo)

# start from CDW product state, expanded through H, then perturbed
psi1 = initialstate_product(H_FCI_shifted, Nx, Ny)

bond_dims(psi) = [dim(left_virtualspace(psi, i)) for i in 1:length(psi)]

"""
    expand_to_saturation(psi, H; chi_max, add, eps, seed, maxrounds)

Repeat `OptimalExpand` + a small perturbation until every bond reaches `chi_max`,
the growth stalls, or `maxrounds` is hit. Returns the state and its environments.
"""
function expand_to_saturation(
        psi, H; chi_max::Int, add::Int, eps::Real, seed::Int = 1234,
        maxrounds::Int = 40, alg_opt = nothing, nsites::Int = 1,
    )
    envs = environments(psi, H, psi)
    energies = Float64[]
    for round in 1:maxrounds
        prev = bond_dims(psi)
        minimum(prev) >= chi_max && break

        psi, envs = changebonds(psi, H, OptimalExpand(; trunc = truncrank(add)), envs)
        # Re-seeded per round so successive perturbations are independent but the
        # whole run stays reproducible.
        psi = perturb_state(psi, eps; seed = seed + round)
        envs = environments(psi, H, psi)

        # VUMPS iterations per round, so the state is actually optimized in the
        # space it was just handed before the next OptimalExpand picks directions
        # from it. Without any, the expansion is state-preserving all the way to
        # chi_max and the state is still energetically the product state when VUMPS
        # takes over -- measured E/site = +1.3e-6 after 14 rounds (FINDINGS 12.3),
        # which is why that route converged worst of the three. One iteration made
        # things *worse* (-0.2492, err 2.7e-1; FINDINGS 12.8), which is what the
        # 1e-3 perturbation and now the second iteration are probing.
        if alg_opt !== nothing
            psi, envs, _ = find_groundstate(psi, H, alg_opt, envs)
        end

        E_site = real(expectation_value(psi, H, envs)) / nsites
        push!(energies, E_site)

        cur = bond_dims(psi)
        @printf("  expand %2d: max chi = %3d, min = %3d, E/site = %+.10f, %s\n",
                round, maximum(cur), minimum(cur), E_site, string(cur))
        cur == prev && (println("  (stalled -- no bond grew)"); break)
    end
    # Trim anything that overshot, so VUMPS starts exactly at chi_max.
    if maximum(bond_dims(psi)) > chi_max
        psi, envs = changebonds(psi, H, SvdCut(; trunc = truncrank(chi_max)), envs)
        @printf("  trimmed to %s\n", string(bond_dims(psi)))
    end
    return psi, envs, energies
end

println("\n=== expansion loop (OptimalExpand + noise + $(vumps_per_round) VUMPS iterations) ===")
alg_round = VUMPS(; maxiter = vumps_per_round, tol = tol_warmup,
                  verbosity = verbosity_warmup, alg_eigsolve = bench_eigsolve(),
                  alg_gauge = bench_gauge(), alg_environments = bench_environments())
t_expand = @elapsed psi1, envs, round_energies = expand_to_saturation(
    psi1, H_FCI_shifted;
    chi_max = chi_warmup, add = expand_add, eps = expand_eps,
    maxrounds = expand_maxrounds, alg_opt = alg_round, nsites = L
)
@printf(
    "expansion: %.2f s, bond dims = %s\n  E/site after expansion = %+.10f\n\n",
    t_expand, string(bond_dims(psi1)),
    real(expectation_value(psi1, H_FCI_shifted)) / L
)

println("=== plain VUMPS on the saturated state ===")
tracker = EnergyTracker(; nsites = L)
alg = VUMPS(;
    maxiter = vumps_maxiter, tol = tol_warmup,
    verbosity = verbosity_warmup, alg_eigsolve = bench_eigsolve(),
    alg_gauge = bench_gauge(), alg_environments = bench_environments(), finalize = tracker
)
t_vumps = @elapsed psi1, envs, err = find_groundstate(psi1, H_FCI_shifted, alg, envs)

E_site = real(expectation_value(psi1, H_FCI_shifted)) / L
@printf("\nVUMPS(%d): %.2f s, final err = %.3e\n", vumps_maxiter, t_vumps, err)
@printf("  E/site   = %+.10f\n", E_site)
@printf("  chi      = %s\n", string(bond_dims(psi1)))
println("  reference: TeNPy -0.2792287344,  IDMRG2 product start -0.2765905800")

# VUMPS uses upstream's 4-argument finalize, so the tracker records no energies
# (they are NaN); per-iteration energies are in the log as `obj = ...`.
times = tracker.times
@printf("\n%-6s%-12s%s\n", "iter", "time [s]", "max chi")
println("-"^34)
for k in eachindex(times)
    @printf("%-6d%-12.3f%d\n", k, times[k], maximum(tracker.bonddims[k]))
end

println("\nper-round E/site during the expansion loop:")
for (r, E) in enumerate(round_energies)
    @printf("  round %2d: %+.10f\n", r, E)
end

npzwrite(
    joinpath(@__DIR__, "meme_trace.npz"),
    Dict("times" => times, "maxchi" => Float64[maximum(b) for b in tracker.bonddims],
         "round_energies" => round_energies)
)
println("\nsaved meme_trace.npz")
