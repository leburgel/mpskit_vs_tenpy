# Locate the exact point where IDMRG2's local eigensolve fails from a chi=1 product
# state, replicating _localupdate_sweep_idmrg2! step by step with MPSKit's real
# solver settings (DynamicTol-adjusted Lanczos tolerance).
using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf

include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel
include(joinpath(@__DIR__, "cdw_state.jl"));   using .CDWState
using MPSKit: AC2, AC2_hamiltonian, transfer_leftenv!, transfer_rightenv!,
    _transpose_front, fixedpoint, updatetol

Nx, Ny = 1, 6
L = Nx * Ny * 2
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
H = shift_my_charge(HaldaneMPO(Nx, Ny, 1.0, t2, 1), Nx, Ny, 1)
Ps = [physicalspace(H, i) for i in 1:L]

# MPSKit's actual first-iteration tolerance: DynamicTol with eps = 1.0
alg_dyn = MPSKit.Defaults.alg_eigsolve()
alg_eig = updatetol(alg_dyn, 0, 1.0)
println("effective eigsolver on iteration 1: ", alg_eig)
println()

psi = cdw_product_state(Ps, [1, 7])
envs = environments(psi, H, psi)
alg_trunc = truncrank(20)

for pos in 1:(L - 1)
    ac2 = AC2(psi, pos; kind = :ACAR)
    h = AC2_hamiltonian(pos, psi, H, psi, envs)
    y = h(ac2)
    @printf(
        "pos %2d: dim=%2d nrm(ac2)=%.3e nrm(H*ac2)=%.3e ",
        pos, dim(space(ac2)), norm(ac2), norm(y)
    )
    local ac2p
    try
        _, ac2p = fixedpoint(h, ac2, :SR, alg_eig)
        @printf("eig OK ")
    catch e
        @printf("\n  !!! EIGSOLVE FAILED at pos %d: %s\n", pos, first(sprint(showerror, e), 90))
        @printf("  ac2   blocks: %s\n", join([string(c, "=>", size(b)) for (c, b) in blocks(ac2)], ", "))
        @printf("  H*ac2 blocks: %s\n", join([string(c, "=>", size(b)) for (c, b) in blocks(y)], ", "))
        @printf("  <ac2|H|ac2> = %s\n", string(dot(ac2, y)))
        break
    end
    al, c, ar = svd_trunc!(ac2p; trunc = alg_trunc)
    @printf("svd: nrm(c)=%.3e ", norm(c))
    normalize!(c)
    psi.AL[pos] = al
    psi.C[pos] = complex(c)
    psi.AR[pos + 1] = _transpose_front(ar)
    psi.AC[pos + 1] = _transpose_front(c * ar)
    transfer_leftenv!(envs, psi, H, psi, pos + 1)
    transfer_rightenv!(envs, psi, H, psi, pos)
    @printf("-> chi=%d\n", dim(left_virtualspace(psi, pos + 1)))
end
