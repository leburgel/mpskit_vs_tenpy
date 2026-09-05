using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

include(joinpath(@__DIR__, "cdw_state.jl"));     using .CDWState
include(joinpath(@__DIR__, "tracked_idmrg.jl")); using .TrackedIDMRG

Nx, Ny = 1, 6; L = 12
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
dens = f_num_3(ComplexF64, U1Irrep)
chain = fill(space(dens, 1), Ny)
occ, vac = 0, 10
H_CDW = InfiniteMPOHamiltonian(chain, [i => (i in (1,7) ? occ : vac) * dens for i in 1:12])
psi_ed = initialstateED(shift_my_charge(H_CDW, Nx, Ny, 1), Nx, Ny)
H = shift_my_charge(HaldaneMPO(Nx, Ny, 1.0, t2, 1), Nx, Ny, 1)

println("=== ED state through the TRACKED driver ===")
println("  initial bonddims: ", bond_dimensions(psi_ed))
alg = IDMRG2(; trunc = truncrank(20), maxiter = 2, tol = 1.0e-4, verbosity = 0)
try
    _, _, _, t = find_groundstate_tracked(psi_ed, H, alg; nsites = L)
    for k in eachindex(t.iters)
        @printf("  iter %d: E/site=%+.8f max_chi=%3d bonddims=%s\n",
                t.iters[k], t.energies[k], maximum(t.bonddims[k]), string(t.bonddims[k]))
    end
    println("  => TRACKED DRIVER WORKS")
catch e
    println("  FAILED: ", first(sprint(showerror, e), 200))
end
