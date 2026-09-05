using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

include(joinpath(@__DIR__, "cdw_state.jl"));     using .CDWState
include(joinpath(@__DIR__, "tracked_idmrg.jl")); using .TrackedIDMRG

println("RandExpand fields: ", fieldnames(MPSKit.RandExpand))

Nx, Ny = 1, 6; L = 12
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
H = shift_my_charge(HaldaneMPO(Nx, Ny, 1.0, t2, 1), Nx, Ny, 1)
Ps = [physicalspace(H, i) for i in 1:L]

for chi_exp in (4, 8, 16)
    println("\n########## RandExpand to chi=$chi_exp ##########")
    try
        Random.seed!(1234)
        psi0 = cdw_product_state(Ps, [1, 7])
        psi_e = changebonds(psi0, MPSKit.RandExpand(; trunc = truncrank(chi_exp)))
        @printf("  after expand: bonddims=%s\n", string(bond_dimensions(psi_e)))
        ns = [real(expectation_value(psi_e, i => occupation_operator(Ps[i]))) for i in 1:L]
        @printf("  <n_i>=%s total=%.6f\n", string(round.(ns, digits=4)), sum(ns))
        @printf("  <H>/site=%.10f\n", real(expectation_value(psi_e, H)) / L)
        alg = IDMRG2(; trunc = truncrank(20) & trunctol(; rtol=1e-10) & truncerror(; atol=1e-14),
                     maxiter = 2, tol = 1.0e-4, verbosity = 0)
        _, _, _, t = find_groundstate_tracked(psi_e, H, alg; nsites = L)
        for k in eachindex(t.iters)
            @printf("  iter %d: E/site=%+.8f max_chi=%3d bonddims=%s\n",
                    t.iters[k], t.energies[k], maximum(t.bonddims[k]), string(t.bonddims[k]))
        end
    catch e
        println("  FAILED: ", first(sprint(showerror, e), 200))
    end
end
