# The shared bond-expansion loop: rounds of (expand -> optimize -> cut) until the
# bond dimensions settle, for either expansion primitive and any truncation.
#
# `trunc` is the *final* truncation -- a rank, a Schmidt-value tolerance, a
# truncation error, or a combination. Each round overshoots on purpose and cuts back
# with `SvdCut(; trunc)`, so with a tolerance the state picks its own bond
# dimensions.
#
# `add` is the per-round overshoot budget. The primitives disagree on what `trunc`
# means, so it is translated:
#
#   OptimalExpand   trunc bounds what is *added*   ->  truncrank(add)
#   VUMPSSvdCut     trunc bounds what is *kept*    ->  truncrank(max chi + add)
#
# the latter clamped to the final rank if there is one.

module Expansion

using LinearAlgebra, Printf, MPSKit, TensorKit, MatrixAlgebraKit
using MatrixAlgebraKit: TruncationStrategy, TruncationByOrder,
    TruncationIntersection, NoTruncation
using MPSKit: Accessors
using ..ProductStart: perturb_state

export bond_dims, expand_to_target

"Left virtual dimension of every bond in the unit cell."
bond_dims(psi) = [dim(left_virtualspace(psi, i)) for i in 1:length(psi)]

# The rank a final truncation imposes, or typemax(Int) if none. Only a rank makes
# the target knowable up front; a tolerance leaves it to the state.
rank_cap(t::TruncationByOrder) = t.howmany
rank_cap(t::TruncationIntersection) = minimum(rank_cap, t.components)
rank_cap(::TruncationStrategy) = typemax(Int)

# A pure rank cut is exact: a no-op unless a bond overshot, and "every bond at the
# target" is already a complete stopping condition.
needs_cut(t::TruncationByOrder, chis) = maximum(chis) > t.howmany
needs_cut(::NoTruncation, chis) = false
needs_cut(::TruncationStrategy, chis) = true
needs_heuristic(::TruncationByOrder) = false
needs_heuristic(::TruncationStrategy) = true

# `trunc` for one round, from the overshoot budget `add` and the current bond dims.
# OptimalExpand's trunc is an increment, everything else's (VUMPSSvdCut, SvdCut) is
# an absolute bound on what is kept.
round_trunc(::OptimalExpand, chis, add, cap) = truncrank(add)
round_trunc(::Any, chis, add, cap) = truncrank(min(cap, maximum(chis) + add))

with_trunc(alg, trunc) = Accessors.@set alg.trunc = trunc

"""
    expand_to_target(psi, H, alg_expand; trunc, add, ...) -> psi, envs, energies, err

Rounds of (expand -> perturb -> optimize -> cut) until the bond dimensions settle or
`maxrounds` is hit.

- `alg_expand`: the expansion primitive, `OptimalExpand` or `VUMPSSvdCut`. Its
  `trunc` field is rebuilt every round from `add`, so the value it carries is
  ignored.
- `trunc`: the final truncation, imposed with `SvdCut` after each round's
  optimization. Bond dimensions never exceed it.
- `add`: the per-round overshoot budget. A bond can grow by less, since
  `OptimalExpand` is capped by the local two-site complement.
- `eps > 0`: perturb after each expansion. `OptimalExpand` needs this -- its added
  directions attach through a zero block, leaving the bond matrices singular.
  `VUMPSSvdCut` does a genuine two-site update and runs with `eps = 0`.
- `alg_opt`: a ground-state algorithm run after each expansion, so the state is
  optimized in the space it was just handed before the next round picks directions
  from it. `err` is its last Galerkin error, `Inf` if there is no optimizer.
- `verbose = false`: silence the per-round report.

Stopping is exact for a rank target -- every bond at the rank, or a round that
changes no bond. With a tolerance the bonds also count as settled once
`|Δchi| <= atol` *and* `|Δchi| <= rtol * |chi|`; requiring both leaves the relative
test to guard the low-chi rounds and the absolute one the converged tail.
"""
function expand_to_target(
        psi, H, alg_expand; trunc::TruncationStrategy, add::Int, eps::Real = 0.0,
        seed::Int = 1234, maxrounds::Int = 40, alg_opt = nothing, nsites::Int = 1,
        atol::Real = 3, rtol::Real = 1.0e-1, verbose::Bool = true,
    )
    envs = environments(psi, H, psi)
    energies = Float64[]
    err = Inf
    cap = rank_cap(trunc)
    for round in 1:maxrounds
        prev = bond_dims(psi)
        if minimum(prev) >= cap
            verbose && println("  (saturated -- every bond at chi = $cap)")
            break
        end

        alg = with_trunc(alg_expand, round_trunc(alg_expand, prev, add, cap))
        psi, envs = changebonds(psi, H, alg, envs)

        # Re-seeded per round: independent draws, still reproducible.
        if eps > 0
            psi = perturb_state(psi, eps; seed = seed + round)
            envs = environments(psi, H, psi)
        end

        if alg_opt !== nothing
            psi, envs, err = find_groundstate(psi, H, alg_opt, envs)
        end

        # Cut the overshoot back, now that the state has been optimized in it.
        if needs_cut(trunc, bond_dims(psi))
            psi, envs = changebonds(psi, H, SvdCut(; trunc), envs)
        end

        E_site = real(expectation_value(psi, H, envs)) / nsites
        push!(energies, E_site)

        cur = bond_dims(psi)
        verbose && @printf(
            "  round %2d: max chi = %3d, min = %3d, E/site = %+.10f, %s%s\n",
            round, maximum(cur), minimum(cur), E_site,
            isfinite(err) ? @sprintf("err = %.3e, ", err) : "", string(cur)
        )

        if cur == prev
            verbose && println("  (stalled -- no bond changed)")
            break
        end
        delta = norm(cur - prev)
        if needs_heuristic(trunc) && delta <= atol && delta <= rtol * norm(cur)
            verbose && @printf("  (settled -- |dchi| = %.3g)\n", delta)
            break
        end
    end
    return psi, envs, energies, err
end

end # module
