using LinearAlgebra, Printf, Plots, NPZ

include(joinpath(@__DIR__, "toolbox", "Toolbox.jl")); using .Toolbox

# Not a controlled variable: BLAS threads make no difference here (FINDINGS 5).
BLAS.set_num_threads(8)

## --- Haldane model ---
t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1
Nx = 1
Ny = 6
L = Nx * Ny * 2

chi_warmup = 100
tol_warmup = 1.0e-4
verbosity_warmup = 4       # > 3 switches on MPSKit's TimerOutputs (stderr)
expand_add = chi_warmup    # per-round bond budget; no budget beyond the target
expand_eps = 1.0e-6        # lifts OptimalExpand's singular bonds (FINDINGS 6)
expand_maxrounds = 40
vumps_per_round = 2        # 2 beat 1 on both energy and error, at the same cost
vumps_maxiter = 50         # the final plain-VUMPS run

t_mpo = @elapsed begin
    H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
    H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)
end
@printf("HaldaneMPO build: %.2f s\n", t_mpo)

psi1 = initialstate_product(H_FCI_shifted, Nx, Ny)

println("\n=== expansion loop (OptimalExpand + noise + $(vumps_per_round) VUMPS iterations) ===")
alg_round = VUMPS(;
    maxiter = vumps_per_round, tol = tol_warmup,
    verbosity = verbosity_warmup, alg_eigsolve = bench_eigsolve(),
    alg_gauge = bench_gauge(), alg_environments = bench_environments()
)
# `trunc` is rebuilt per round from `expand_add`; for OptimalExpand that is the
# identity, since its trunc is already a per-bond increment.
alg_expand = OptimalExpand(; trunc = truncrank(expand_add))
t_expand = @elapsed psi1, envs, round_energies, _ = expand_to_target(
    psi1, H_FCI_shifted, alg_expand;
    trunc = truncrank(chi_warmup), add = expand_add, eps = expand_eps,
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

# VUMPS's 4-argument finalize hands over no energy, so the tracker records NaN;
# per-iteration energies are in the stderr log as `obj = ...`.
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
    joinpath(@__DIR__, "optexpand_vumps_trace.npz"),
    Dict(
        "times" => times, "maxchi" => Float64[maximum(b) for b in tracker.bonddims],
        "round_energies" => round_energies
    )
)
println("\nsaved optexpand_vumps_trace.npz")
