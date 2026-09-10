using LinearAlgebra, Printf, Plots, NPZ

include(joinpath(@__DIR__, "toolbox", "Toolbox.jl")); using .Toolbox

# Not a controlled variable: BLAS threads make no difference here (FINDINGS 5).
BLAS.set_num_threads(8)
# MPSKit picks a DynamicScheduler whenever nthreads > 1, so `julia -t 12` is all
# unit-cell parallelism needs. Logged so the run records which regime it was in.
println("Julia threads: ", Threads.nthreads(),
        ";  MPSKit scheduler: ", MPSKit.Defaults.scheduler[])
## --- Haldane model ---
t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1
Nx = 1
Ny = 6   # cylinder circumference, in unit cells
L = Nx * Ny * 2

H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)

psi1 = initialstate_product(H_FCI_shifted, Nx, Ny)

chi_warmup = 100
tol_warmup = 1.0e-4
verbosity_warmup = 4     # > 3 switches on MPSKit's TimerOutputs (stderr)
maxiter_warmup = 50      # ceiling on expansion rounds; ~7 is what it takes
expand_add = chi_warmup  # per-round bond budget; no budget beyond the target
vumps_per_round = 2      # 2 beat 1 on both energy and error, at the same cost
vumps_maxiter = 50       # the final plain-VUMPS run

tracker = EnergyTracker(; nsites = L)

# One-site VUMPS plus explicit bond expansion, in place of a single IDMRG2 call.
# VUMPSSvdCut does a genuine two-site update, so bonds grow and the state stays
# non-singular -- no perturbation needed, unlike the OptimalExpand route.
alg = VUMPS(; maxiter = vumps_per_round, tol = tol_warmup,
            verbosity = verbosity_warmup, alg_eigsolve = bench_eigsolve(),
            alg_gauge = bench_gauge(), alg_environments = bench_environments(),
            finalize = tracker)
# changebonds algorithms use static tolerances upstream; keep that. `trunc` is
# rebuilt per round from `expand_add`, so the value here is a placeholder.
alg_expand = VUMPSSvdCut(;
    trunc = truncrank(chi_warmup),
    alg_eigsolve = bench_eigsolve(; dynamic_tols = false),
    alg_gauge = bench_gauge(; dynamic_tols = false),
)

println("\n=== expansion loop ($(vumps_per_round) VUMPS iterations per expansion) ===")
t_warmup = @elapsed psi1, envs, round_energies, err = expand_to_target(
    psi1, H_FCI_shifted, alg_expand;
    trunc = truncrank(chi_warmup), add = expand_add, maxrounds = maxiter_warmup,
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

# Bonds at chi_warmup: hand over to plain VUMPS to converge.
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

# VUMPS's 4-argument finalize hands over no energy, so the tracker records NaN;
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
savefig(p, joinpath(@__DIR__, "vumpssvd_vumps_energy.png"))

npzwrite(
    joinpath(@__DIR__, "vumpssvd_vumps_trace.npz"),
    Dict(
        "times" => times,
        "maxchi" => Float64[maximum(b) for b in tracker.bonddims],
        "minchi" => Float64[minimum(b) for b in tracker.bonddims],
        "round_energies" => round_energies,
        "E_final" => [E_final],
    )
)
println("\nsaved vumpssvd_vumps_trace.npz")
