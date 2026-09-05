# Why does the bond dimension not grow during the left-to-right IDMRG2 leg
# when starting from a chi=1 product state?
using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel
include(joinpath(@__DIR__, "cdw_state.jl"));   using .CDWState
using MPSKit: AC2, AC2_hamiltonian, fixedpoint, updatetol

Nx, Ny = 1, 6
L = 12
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * acos(3 * sqrt(3 / 43)))
H = shift_my_charge(HaldaneMPO(Nx, Ny, 1.0, t2, 1), Nx, Ny, 1)
Ps = [physicalspace(H, i) for i in 1:L]
psi = cdw_product_state(Ps, [1, 7])
envs = environments(psi, H, psi)
alg_eig = updatetol(MPSKit.Defaults.alg_eigsolve(), 0, 1.0)

for pos in (1, 2, 3)
    println("\n########## pos $pos ##########")
    ac2 = AC2(psi, pos; kind = :ACAR)
    println("  Vl = ", left_virtualspace(psi, pos))
    println("  P$pos = ", Ps[pos])
    println("  P$(pos + 1) = ", Ps[pos + 1])
    println("  ac2 space = ", space(ac2))
    println("  coupled blocks of ac2 (sector => blocksize):")
    for (c, b) in blocks(ac2)
        @printf("     %-48s %s\n", string(c), string(size(b)))
    end

    h = AC2_hamiltonian(pos, psi, H, psi, envs)
    _, ac2p = fixedpoint(h, ac2, :SR, alg_eig)
    println("  coupled blocks of optimized ac2' :")
    for (c, b) in blocks(ac2p)
        @printf("     %-48s %s  norm=%.6e\n", string(c), string(size(b)), norm(b))
    end

    # full (untruncated) SVD spectrum, per coupled sector
    u, s, v = svd_trunc!(copy(ac2p); trunc = notrunc())
    println("  FULL svd singular values per sector:")
    for (c, b) in blocks(s)
        @printf("     %-48s %s\n", string(c), string(round.(diag(b), digits = 12)))
    end
    println("  => full   bond space: ", space(s, 1))
    ut, st, vt = svd_trunc!(copy(ac2p); trunc = truncrank(20))
    println("  => truncrank(20) bond space: ", space(st, 1), "  (dim ", dim(space(st, 1)), ")")
end
