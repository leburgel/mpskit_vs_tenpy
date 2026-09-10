using JLD2, LinearAlgebra
using Dates, TOML
using Plots
using Logging, Printf, NPZ

include(joinpath(@__DIR__, "toolbox", "Toolbox.jl")); using .Toolbox

# BLAS threads make no difference on this problem (FINDINGS 5), so this is not part
# of what the benchmark controls -- just a sane default.
BLAS.set_num_threads(8)
# Unit-cell parallelism is implicit: MPSKit picks a DynamicScheduler whenever
# Threads.nthreads() > 1, so `julia -t 12` is all that is needed. Logged so the run
# records which regime it was in.
println("Julia threads: ", Threads.nthreads(),
        ";  MPSKit scheduler: ", MPSKit.Defaults.scheduler[])
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
# 4 not 3: MPSKit only builds a live TimerOutput when verbosity > 3, which is what
# produces the per-stage split (on stderr). At 3 the timers cost nothing.

# Ceiling on rounds of (expand -> VUMPS); the loop stops once every bond is at
# chi_warmup or a round grows nothing. A bond at best doubles per call, so ~8
# rounds is what it actually takes.
maxiter_warmup = 50

# How much a bond may gain per round. chi_warmup means no budget beyond the target
# itself, which is what VUMPSSvdCut was given by hand before the loop was shared.
expand_add = chi_warmup

# VUMPS iterations between successive changebonds calls. VUMPSSvdCut picks its new
# directions from the current state, so a better-optimized state expands better:
# 2 beat 1 on both energy and error at the same warm-up cost.
vumps_per_round = 2

# Iterations for the final plain-VUMPS convergence run.
vumps_maxiter = 50

tracker = EnergyTracker(; nsites = L)

# One-site VUMPS plus explicit bond expansion, in place of a single IDMRG2 call.
# VUMPSSvdCut rather than OptimalExpand: it does a genuine two-site update and
# truncates to `trunc` itself, so bonds grow and the state stays non-singular, where
# OptimalExpand is confined to the null space and needs the perturbation of
# mpskit_optexpand_vumps.jl. Both routes share `expand_to_target`.
alg = VUMPS(; maxiter = vumps_per_round, tol = tol_warmup,
            verbosity = verbosity_warmup, alg_eigsolve = bench_eigsolve(),
            alg_gauge = bench_gauge(), alg_environments = bench_environments(),
            finalize = tracker)
# changebonds algorithms use static tolerances upstream; keep that, and take only
# the krylovdim/maxiter/eager part. `trunc` is rebuilt per round by expand_to_target
# from `expand_add`, so the value here is a placeholder.
alg_expand = VUMPSSvdCut(;
    trunc = truncrank(chi_warmup),
    alg_eigsolve = bench_eigsolve(; dynamic_tols = false),
    alg_gauge = bench_gauge(; dynamic_tols = false),
)

println("\n=== expansion loop ($(vumps_per_round) VUMPS iterations per expansion) ===")
t_warmup = @elapsed psi1, envs, round_energies, err = expand_to_target(
    psi1, H_FCI_shifted, alg_expand;
    chi_max = chi_warmup, add = expand_add, maxrounds = maxiter_warmup,
    alg_opt = alg, nsites = L
)
@printf(
    "\nafter expansion loop: %.2f s, err = %.3e, bond dims = %s\n",
    t_warmup, err, string(bond_dims(psi1))
)
@printf(
    "  all bonds at chi = %d: %s\n", chi_warmup,
    minimum(bond_dims(psi1)) >= chi_warmup ? "YES" : "NO"
)

# Once the bonds are at chi_warmup, hand the state to plain VUMPS to converge.
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

# Tracker energies are NaN for VUMPS, so this column is time and bond dimension;
# per-iteration energies are in the stderr log as `obj = ...`.
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
savefig(p, joinpath(@__DIR__, "vumps_expand_energy.png"))

npzwrite(
    joinpath(@__DIR__, "vumps_expand_trace.npz"),
    Dict(
        "times" => times,
        "maxchi" => Float64[maximum(b) for b in tracker.bonddims],
        "minchi" => Float64[minimum(b) for b in tracker.bonddims],
        "round_energies" => round_energies,
        "E_final" => [E_final],
    )
)
println("\nsaved vumps_expand_trace.npz")
