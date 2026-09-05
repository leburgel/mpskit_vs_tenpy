using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

## ===== SMOKE TEST: product state + tracked IDMRG2 + composed truncation =====
include(joinpath(@__DIR__, "cdw_state.jl"));    using .CDWState
include(joinpath(@__DIR__, "tracked_idmrg.jl")); using .TrackedIDMRG

Nx, Ny = 1, 6
L = Nx * Ny * 2
t1 = 1.0
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
V1 = 1

H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
H = shift_my_charge(H_FCI, Nx, Ny, 1)
Ps = [physicalspace(H, i) for i in 1:L]

# TeNPy's cdw_pattern under order="Cstyle" occupies A(y=0) and A(y=3),
# which is Julia's sites 1 and 7.
psi0 = cdw_product_state(Ps, [1, 7])
@printf("initial bond dims: %s\n", string(bond_dimensions(psi0)))
ns = [real(expectation_value(psi0, i => occupation_operator(Ps[i]))) for i in 1:L]
@printf("initial <n_i>: %s  (total %.6f)\n", string(round.(ns, digits = 6)), sum(ns))
@printf("initial <H>/site: %.10f\n\n", real(expectation_value(psi0, H)) / L)

trunc = truncrank(20) & trunctol(; rtol = 1.0e-10) & truncerror(; atol = 1.0e-14)
alg = IDMRG2(; trunc, maxiter = 3, tol = 1.0e-4, verbosity = 1)
psi, envs, err, tr = find_groundstate_tracked(psi0, H, alg; nsites = L)

println("iter  time[s]   E/site          err        max_chi  bond dims")
println("-"^78)
for k in eachindex(tr.iters)
    @printf(
        "%4d  %7.2f  %+.10f  %9.3e  %5d   %s\n",
        tr.iters[k], tr.times[k], tr.energies[k], tr.errors[k],
        maximum(tr.bonddims[k]), string(tr.bonddims[k])
    )
end
