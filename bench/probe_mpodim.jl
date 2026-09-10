# Does the `lb/smaller_mpos` MPOHamiltonian constructor ("De-duplicate channels
# where possible", 809c8109 on top of the 4b579944 baseline) narrow MPSKit's MPO?
#
# Baseline, FINDINGS.md 11.4: MPSKit 82 on every bond at Ny=6, TeNPy 37 (Cstyle,
# the like-for-like ordering). Ny=3 baseline, measured 2026-09-08: 49 per bond.
using Printf, LinearAlgebra
include(joinpath(@__DIR__, "..", "toolbox", "Toolbox.jl")); using .Toolbox
BLAS.set_num_threads(8)
@printf("MPSKit %s   TensorKit %s\n", pkgversion(MPSKit), pkgversion(TensorKit))

t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1

for Ny in (3, 6)
    L = 2Ny
    t = @elapsed H = HaldaneMPO(1, Ny, t1, t2, V1)
    Hs = shift_my_charge(H, 1, Ny, 1)
    dims_raw = [dim(left_virtualspace(H, n)) for n in 1:L]
    dims_shift = [dim(left_virtualspace(Hs, n)) for n in 1:L]
    @printf("\nNy = %d (%d sites), build %.2f s\n", Ny, L, t)
    @printf("  raw     : max %3d  mean %5.1f  %s\n",
            maximum(dims_raw), sum(dims_raw)/L, string(dims_raw))
    @printf("  shifted : max %3d  mean %5.1f  %s\n",
            maximum(dims_shift), sum(dims_shift)/L, string(dims_shift))
end
println("\nbaseline (4b579944): Ny=3 -> 49 per bond;  Ny=6 -> 82 per bond")
println("TeNPy Cstyle, Ny=6 : 37 (interior), tapering to 32 at the cell edges")
