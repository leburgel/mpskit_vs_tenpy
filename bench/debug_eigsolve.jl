using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

include(joinpath(@__DIR__, "cdw_state.jl")); using .CDWState
using MPSKit: AC2, AC2_hamiltonian, fixedpoint

Nx, Ny = 1, 6; L = 12
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
H = shift_my_charge(HaldaneMPO(Nx, Ny, 1.0, t2, 1), Nx, Ny, 1)
Ps = [physicalspace(H, i) for i in 1:L]

for (label, psi) in (("PRODUCT (chi=1)", cdw_product_state(Ps, [1, 7])),)
    println("\n########## $label ##########")
    envs = environments(psi, H, psi)
    for pos in 1:3
        ac2 = AC2(psi, pos; kind = :ACAR)
        h = AC2_hamiltonian(pos, psi, H, psi, envs)
        y = h(ac2)
        @printf("pos %d: dim(two-site space)=%d  norm(ac2)=%.6e  norm(H*ac2)=%.6e\n",
                pos, dim(space(ac2)), norm(ac2), norm(y))
        @printf("        <ac2|H|ac2> = %s\n", string(dot(ac2, y)))
        @printf("        any NaN in H*ac2: %s   any Inf: %s\n",
                any(isnan, y.data), any(isinf, y.data))
        # sector-resolved dimensions of the two-site tensor
        @printf("        blocks: %s\n",
                join([string(c, "=>", size(b)) for (c, b) in blocks(ac2)], ", "))
        try
            vals, vecs, info = eigsolve(h, ac2, 1, :SR,
                Lanczos(; krylovdim = 30, maxiter = 200, tol = 1e-10, eager = true))
            @printf("        eigsolve OK: E=%s converged=%d numiter=%d\n",
                    string(vals[1]), info.converged, info.numiter)
        catch e
            @printf("        eigsolve FAILED: %s\n", first(sprint(showerror, e), 90))
        end
    end
end
