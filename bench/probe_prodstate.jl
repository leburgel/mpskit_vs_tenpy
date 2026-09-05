using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit, Printf
include(joinpath(@__DIR__, "..", "model.jl")); using .HaldaneModel

## ===== PROBE: build a bond-dim-1 CDW product state directly (no ED) =====

Nx, Ny = 1, 6
L = Nx * Ny * 2

# Reproduce the charge-shifted physical spaces without running ED.
dens = f_num_3(ComplexF64, U1Irrep)
chain = fill(space(dens, 1), Ny)
H_CDW = InfiniteMPOHamiltonian(chain, [i => 1.0 * dens for i in 1:12])
H_CDW_shifted = shift_my_charge(H_CDW, Nx, Ny, 1)
Ps = [physicalspace(H_CDW_shifted, i) for i in 1:L]
println("physical spaces (charge-shifted):")
for i in 1:L; println("  site $i: ", Ps[i]); end

u1charge(s) = s.sectors[2].charge
occ_sector(P) = argmax(u1charge, collect(sectors(P)))
emp_sector(P) = argmin(u1charge, collect(sectors(P)))

"""
Bond-dimension-1 InfiniteMPS product state with electrons on `occupied` sites.
Virtual charges accumulate the physical charges; the unit cell must close.
"""
function cdw_product_state(Ps, occupied)
    L = length(Ps)
    I = sectortype(first(Ps))
    psecs = [i in occupied ? occ_sector(Ps[i]) : emp_sector(Ps[i]) for i in 1:L]
    # accumulate virtual charges: c_{i+1} = c_i x p_i  (abelian -> unique channel)
    cs = Vector{I}(undef, L + 1)
    cs[1] = one(I)
    for i in 1:L
        cs[i + 1] = first(cs[i] ⊗ psecs[i])
    end
    @printf("  virtual charges: closes = %s  (c_1 = %s, c_%d = %s)\n",
            cs[1] == cs[L + 1], string(cs[1]), L + 1, string(cs[L + 1]))
    cs[1] == cs[L + 1] || error("unit cell does not close: net charge $(cs[L+1])")
    Vs = [Vect[I](cs[i] => 1) for i in 1:(L + 1)]
    As = map(1:L) do i
        A = zeros(ComplexF64, Vs[i] ⊗ Ps[i] ← Vs[i + 1])
        block(A, cs[i + 1]) .= 1
        return A
    end
    return InfiniteMPS(As)
end

# TeNPy (Cstyle order) occupies the first A and first B site => Julia sites 1 and 2
for occupied in ([1, 2], [1, 7])
    println("\n=== product state with electrons on sites $occupied ===")
    psi = cdw_product_state(Ps, occupied)
    bds = [dim(left_virtualspace(psi, i)) for i in 1:L]
    println("  bond dims: ", bds, "  max = ", maximum(bds))
    nop(P) = (n = zeros(ComplexF64, P ← P); block(n, occ_sector(P)) .= 1; n)
    ns = [real(expectation_value(psi, i => nop(Ps[i]))) for i in 1:L]
    println("  <n_i>: ", round.(ns, digits = 8), "  total = ", round(sum(ns), digits = 8))
    println("  norm = ", norm(psi))
end

println("\n=== truncation strategy composition ===")
for expr in ("truncrank(100)", "truncrank(100) & trunctol(1e-10)", "trunctol(1e-10)")
    try
        v = eval(Meta.parse(expr))
        println("  $expr  =>  ", v)
    catch e
        println("  $expr  =>  FAILED: ", first(sprint(showerror, e), 120))
    end
end
