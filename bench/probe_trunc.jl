using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

include(joinpath(@__DIR__, "cdw_state.jl"));     using .CDWState
include(joinpath(@__DIR__, "tracked_idmrg.jl")); using .TrackedIDMRG

Nx, Ny = 1, 6; L = 12
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
H = shift_my_charge(HaldaneMPO(Nx, Ny, 1.0, t2, 1), Nx, Ny, 1)
Ps = [physicalspace(H, i) for i in 1:L]
psi0 = cdw_product_state(Ps, [1, 7])

cases = [
    ("truncrank(20) only",              truncrank(20)),
    ("truncrank(20) & trunctol",        truncrank(20) & trunctol(; rtol = 1.0e-10)),
    ("truncrank(20) & truncerror",      truncrank(20) & truncerror(; atol = 1.0e-14)),
]
for (name, tr_alg) in cases
    println("\n########## $name ##########")
    try
        alg = IDMRG2(; trunc = tr_alg, maxiter = 2, tol = 1.0e-4, verbosity = 0)
        _, _, _, t = find_groundstate_tracked(psi0, H, alg; nsites = L)
        for k in eachindex(t.iters)
            @printf("  iter %d: E/site=%+.8f  max_chi=%3d  bonddims=%s\n",
                    t.iters[k], t.energies[k], maximum(t.bonddims[k]), string(t.bonddims[k]))
        end
    catch e
        println("  FAILED: ", first(sprint(showerror, e), 200))
    end
end
