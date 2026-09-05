# Deterministic bond-dimension-1 CDW product state for the charge-shifted Haldane model.
#
# Replaces `initialstateED`, which (a) cost ~90 s, (b) used an unseeded `rand!`
# start vector inside `exact_diagonalization` and so gave a different state every
# run, and (c) produced a state with bond dimensions 1,2,4,5,6,7,8,7,6,5,4,2 --
# whereas TeNPy starts from a genuine product state with all bonds 1.

module CDWState

using MPSKit, TensorKit

export cdw_product_state, occupation_operator, bond_dimensions

# Of the two sectors in a charge-shifted physical space, the occupied one carries
# the larger U(1) charge (they differ by the electron charge 3).
u1charge(s) = s.sectors[2].charge
occ_sector(P) = argmax(u1charge, collect(sectors(P)))
emp_sector(P) = argmin(u1charge, collect(sectors(P)))

"""
    occupation_operator(P)

Number operator on a (charge-shifted) physical space `P`.
"""
function occupation_operator(P)
    n = zeros(ComplexF64, P ← P)
    block(n, occ_sector(P)) .= 1
    return n
end

bond_dimensions(mps) = [dim(left_virtualspace(mps, i)) for i in 1:length(mps)]

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

end # module
