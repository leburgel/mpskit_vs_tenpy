# The shared bond-expansion loop: grow every bond to chi_max in rounds of
# (expand -> optimize), for either expansion primitive.
#
# The two primitives disagree on what `trunc` means, so the loop takes `add` -- how
# much a bond may gain per round -- and translates it:
#
#   OptimalExpand   trunc bounds what is *added*   ->  truncrank(add)
#   VUMPSSvdCut     trunc bounds what is *kept*    ->  truncrank(max chi + add)
#
# the latter clamped to chi_max, the former left to the final trim (a scalar trunc
# cannot cap overshoot on the leading bond without also starving the laggard). With
# add >= chi_max both reduce to what the scripts passed by hand before this merge.

module Expansion

using Printf, MPSKit, TensorKit, MatrixAlgebraKit
using MPSKit: Accessors
using ..ProductStart: perturb_state

export bond_dims, expand_to_target

"""
    bond_dims(psi) -> Vector{Int}

Left virtual dimension of every bond in the unit cell.
"""
bond_dims(psi) = [dim(left_virtualspace(psi, i)) for i in 1:length(psi)]

# `trunc` for one round, from the per-round budget `add` and the current bond dims.
# OptimalExpand's trunc is an increment, everything else's (VUMPSSvdCut, SvdCut) is
# an absolute bound on what is kept.
round_trunc(::OptimalExpand, chis, add, chi_max) = truncrank(add)
round_trunc(::Any, chis, add, chi_max) = truncrank(min(chi_max, maximum(chis) + add))

with_trunc(alg, trunc) = Accessors.@set alg.trunc = trunc

"""
    expand_to_target(psi, H, alg_expand; chi_max, add, ...) -> psi, envs, energies, err

Rounds of (expand -> perturb -> optimize) until every bond reaches `chi_max`, the
growth stalls, or `maxrounds` is hit.

`alg_expand` is the expansion primitive -- `OptimalExpand` or `VUMPSSvdCut`. Its
`trunc` field is rebuilt every round from `add` (see the header), so whatever it is
constructed with is ignored.

`add` is how much each bond may gain per round; it defaults to `chi_max`, i.e. no
per-round budget beyond the target itself. A bond can still grow by less, since
`OptimalExpand` is capped by the local two-site complement.

`eps > 0` perturbs the state after each expansion. `OptimalExpand` needs it: the
added directions attach through a zero block, leaving the bond matrices singular.
`VUMPSSvdCut` does a genuine two-site update and needs no perturbation, so it runs
with `eps = 0`.

`alg_opt`, if given, is a ground-state algorithm run after each expansion, so the
state is optimized in the space it was just handed before the next round picks its
directions from it. Without it nothing optimizes the state between expansions.
`err` is that optimizer's last Galerkin error, `Inf` if there is none.
"""
function expand_to_target(
        psi, H, alg_expand; chi_max::Int, add::Int = chi_max, eps::Real = 0.0,
        seed::Int = 1234, maxrounds::Int = 40, alg_opt = nothing, nsites::Int = 1,
    )
    envs = environments(psi, H, psi)
    energies = Float64[]
    err = Inf
    for round in 1:maxrounds
        prev = bond_dims(psi)
        if minimum(prev) >= chi_max
            println("  (saturated -- every bond at chi = $chi_max)")
            break
        end

        alg = with_trunc(alg_expand, round_trunc(alg_expand, prev, add, chi_max))
        psi, envs = changebonds(psi, H, alg, envs)

        # Re-seeded per round, so successive perturbations are independent but the
        # whole run stays reproducible.
        if eps > 0
            psi = perturb_state(psi, eps; seed = seed + round)
            envs = environments(psi, H, psi)
        end

        if alg_opt !== nothing
            psi, envs, err = find_groundstate(psi, H, alg_opt, envs)
        end

        E_site = real(expectation_value(psi, H, envs)) / nsites
        push!(energies, E_site)

        cur = bond_dims(psi)
        @printf(
            "  round %2d: max chi = %3d, min = %3d, E/site = %+.10f, %s%s\n",
            round, maximum(cur), minimum(cur), E_site,
            isfinite(err) ? @sprintf("err = %.3e, ", err) : "", string(cur)
        )
        cur == prev && (println("  (stalled -- no bond grew)"); break)
    end
    # Trim anything that overshot, so the optimizer starts exactly at chi_max.
    if maximum(bond_dims(psi)) > chi_max
        psi, envs = changebonds(psi, H, SvdCut(; trunc = truncrank(chi_max)), envs)
        @printf("  trimmed to %s\n", string(bond_dims(psi)))
    end
    return psi, envs, energies, err
end

end # module
