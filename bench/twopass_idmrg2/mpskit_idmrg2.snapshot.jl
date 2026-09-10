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
Ny = 6   # cylinder circumference, in unit cells
H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)

psi1 = initialstate_product(H_FCI_shifted, Nx, Ny)

chi_warmup = 100
tol_warmup = 1.0e-4
maxiter_warmup = 50
# > 3 also switches on MPSKit's TimerOutputs: 4 for the per-stage split (on
# stderr), 3 for clean timings.
verbosity_warmup = parse(Int, get(ENV, "BENCH_VERBOSITY", "4"))

eigsolve_alg = bench_eigsolve()
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

# Tracked energies are IDMRG2's ΔE estimator, not a variational bound (FINDINGS 7),
# so print both. `envs` was just returned, so this costs no rebuild.
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
