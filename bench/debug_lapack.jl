using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

include(joinpath(@__DIR__, "cdw_state.jl")); using .CDWState

Nx, Ny = 1, 6; L = 12
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
H = shift_my_charge(HaldaneMPO(Nx, Ny, 1.0, t2, 1), Nx, Ny, 1)
Ps = [physicalspace(H, i) for i in 1:L]
psi0 = cdw_product_state(Ps, [1, 7])

try
    find_groundstate(psi0, H, IDMRG2(; trunc = truncrank(20), maxiter = 1, tol = 1e-4, verbosity = 0))
catch e
    bt = catch_backtrace()
    println("### ERROR: ", sprint(showerror, e))
    println("### innermost frames (file:line):")
    for (i, f) in enumerate(stacktrace(bt))
        i > 22 && break
        @printf("%3d  %-38s %s:%s\n", i, string(f.func), basename(string(f.file)), f.line)
    end
end
