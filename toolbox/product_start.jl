# The benchmark's initial state: a deterministic chi=1 CDW product state, plus just
# enough expansion and noise that IDMRG2 can start from it. `initialstate_product`
# in Toolbox.jl assembles the two steps.
#
# IDMRG2 cannot start from a bare chi=1 product state: at 1/6 filling the symmetry
# locks most two-site blocks to a single state, so Lanczos gets an exhausted Krylov
# space and `stegr!` fails, and IDMRG2 has no mixer of its own. Two steps fix it:
#
#   1. One round of `OptimalExpand` (via `expand_to_target`), which builds the
#      update *through H* -- unlike `RandExpand`, which stalls at chi=2 on a
#      charge-locked state. One round reaches chi=2 and IDMRG2 takes it from there.
#   2. A small symmetric perturbation, since `OptimalExpand` attaches its new
#      directions through a zero block and leaves the bond matrices singular.
#      eps = 1e-2..1e-6 all work and agree on the converged energy.
#
# This replaced an exact-diagonalization start, which was slow, nondeterministic
# (unseeded `rand!`) and did not begin at chi=1 the way TeNPy does. Seeded, and
# confirmed bit-reproducible across runs.

module ProductStart

using LinearAlgebra, Random, MPSKit, TensorKit

export cdw_product_state, perturb_state

# Of the two sectors in a charge-shifted physical space, the occupied one carries
# the larger U(1) charge (they differ by the electron charge 3).
u1charge(s) = s.sectors[2].charge
occ_sector(P) = argmax(u1charge, collect(sectors(P)))
emp_sector(P) = argmin(u1charge, collect(sectors(P)))

"""
    cdw_product_state(Ps, occupied)

Bond-dimension-1 `InfiniteMPS` product state on physical spaces `Ps` with electrons
on the sites in `occupied`. Virtual charges accumulate the physical charges; the
unit cell must close (net charge zero), which is what `add_physical_charge`
arranges for the chosen filling.
"""
function cdw_product_state(Ps, occupied)
    L = length(Ps)
    I = sectortype(first(Ps))
    psecs = [i in occupied ? occ_sector(Ps[i]) : emp_sector(Ps[i]) for i in 1:L]

    cs = Vector{I}(undef, L + 1)
    cs[1] = one(I)
    for i in 1:L
        cs[i + 1] = first(cs[i] ⊗ psecs[i])
    end
    cs[1] == cs[L + 1] || error(
        "unit cell does not close: net charge $(cs[L + 1]) after $L sites. " *
        "Check that `occupied` matches the filling used in add_physical_charge."
    )

    Vs = [Vect[I](cs[i] => 1) for i in 1:(L + 1)]
    As = map(1:L) do i
        A = zeros(ComplexF64, Vs[i] ⊗ Ps[i] ← Vs[i + 1])
        block(A, cs[i + 1]) .= 1
        return A
    end
    return InfiniteMPS(As)
end

"""
    perturb_state(psi, eps = 1e-6; seed = 1234) -> InfiniteMPS

Add a symmetry-respecting random perturbation of relative size `eps` to every
tensor and re-gauge, making the bond matrices full rank.

`randn!` fills only the charge blocks the tensor already allows, so filling and
symmetry sectors are unchanged. Seeded, so the result is reproducible.
"""
function perturb_state(psi, eps::Real = 1.0e-6; seed::Int = 1234)
    Random.seed!(seed)
    As = map(1:length(psi)) do i
        A = copy(psi.AL[i])
        N = randn!(similar(A))
        return A + (eps * norm(A) / norm(N)) * N
    end
    return InfiniteMPS(As)
end

end # module
