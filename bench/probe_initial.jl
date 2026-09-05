using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

## ============ PROBE: what does initialstateED actually produce? ============

Nx, Ny = 1, 6
t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1

occ, vac = 0, 10
dens = f_num_3(ComplexF64, U1Irrep)
chain = fill(space(dens, 1), Ny)
ops = [i => (i in (1, 7) ? occ : vac) * dens for i in 1:12]
H_CDW = InfiniteMPOHamiltonian(chain, ops)
H_CDW_shifted = shift_my_charge(H_CDW, Nx, Ny, 1)

@printf("length(chain)=%d  length(H_CDW)=%d  length(H_CDW_shifted)=%d\n",
        length(chain), length(H_CDW), length(H_CDW_shifted))

psi1 = initialstateED(H_CDW_shifted, Nx, Ny)
@printf("length(psi1) = %d\n\n", length(psi1))

println("=== bond dimensions of initial InfiniteMPS ===")
bds = [dim(left_virtualspace(psi1, i)) for i in 1:length(psi1)]
println("  ", bds, "   max = ", maximum(bds))

# Build the number operator directly in each site's (charge-shifted) physical space:
# of the two sectors present, the one with the LARGER U1 charge is the occupied one.
function number_op(V)
    n = zeros(ComplexF64, V ← V)
    secs = collect(sectors(V))
    u1(s) = s.sectors[2].charge
    occ_sec = argmax(u1, secs)
    block(n, occ_sec) .= 1
    return n
end

println("\n=== per-site occupation <n_i> ===")
total = 0.0
for i in 1:length(psi1)
    V = physicalspace(psi1, i)
    ni = real(expectation_value(psi1, i => number_op(V)))
    global total += ni
    @printf("  site %2d: %+.10f   (space %s)\n", i, ni, string(V))
end
@printf("  => electrons per unit cell = %.8f   filling = %.8f /site\n", total, total / length(psi1))

H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)
@printf("\n<H_FCI>/site  = %.10f\n", real(expectation_value(psi1, H_FCI_shifted)) / 12)
@printf("<H_CDW>       = %.10f  (0 means electrons sit exactly on the CDW sites)\n",
        real(expectation_value(psi1, H_CDW_shifted)))
