# Energy-vs-walltime tracking for MPSKit solvers, via the `finalize` callback.

module EnergyTracking

using MPSKit, TensorKit
using ..Expansion: bond_dims

export EnergyTracker, reset!

"""
    EnergyTracker(; nsites = 12, inner = nothing)

A `finalize` callback recording per-iteration wall time, energy per site, Galerkin
error and the per-bond dimension profile. Pass it as the `finalize` field of any
algorithm that has one:

    tracker = EnergyTracker(; nsites = 12)
    psi, envs, err = find_groundstate(psi, H, IDMRG2(; trunc, finalize = tracker))

Energy and error are handed over by the algorithm, not computed here, so tracking
is free (570 μs across a 279 s solve). That needs the local 6-argument
`finalize(iter, ψ, H, envs, ϵ, ΔE)` patch; on upstream's 4-argument contract
(VUMPS, DMRG, ...) both are `NaN`. Recomputing them here is not an option: IDMRG
hands over propagated environments, so an absolute `expectation_value` against them
carries an extensive offset, and rebuilding costs ~18 s per iteration at chi=100.

`energies` is `ΔE / nsites`, iDMRG's energy-per-unit-cell estimator. `times` is
algorithm-only, with this tracker's own cost subtracted (per-iteration overhead is
in `overheads`). `iters` restarts at 0 on every `find_groundstate` call, so use
`eachindex` or `steps` as an x-axis across repeated solver calls.
"""
mutable struct EnergyTracker{F}
    iters::Vector{Int}
    steps::Vector{Int}
    times::Vector{Float64}
    energies::Vector{Float64}
    errors::Vector{Float64}
    bonddims::Vector{Vector{Int}}
    overheads::Vector{Float64}
    nsites::Int
    t0::Float64
    overhead_total::Float64
    inner::F
end
function EnergyTracker(; nsites::Int = 12, inner = nothing)
    return EnergyTracker(
        Int[], Int[], Float64[], Float64[], Float64[], Vector{Int}[], Float64[],
        nsites, time(), 0.0, inner
    )
end

# Upstream's 4-argument contract (VUMPS/DMRG/DMRG2/VOMPS/TDVP): no energy or error
# is passed, so both go down as NaN. Times and bond dimensions are unaffected.
(tr::EnergyTracker)(iter, psi, H, envs) = tr(iter, psi, H, envs, NaN, NaN)

function (tr::EnergyTracker)(iter, psi, H, envs, eps, dE)
    t_entry = time()
    # `inner` chains a real finalizer behind the recording, so tracking never
    # displaces an actual callback.
    if tr.inner !== nothing
        psi, envs = tr.inner(iter, psi, H, envs, eps, dE)
    end
    push!(tr.iters, iter)
    push!(tr.steps, length(tr.steps) + 1)
    # Algorithm-only: wall clock at entry, minus everything this tracker has spent
    # so far -- otherwise each timestamp carries all previous iterations' overhead.
    push!(tr.times, (t_entry - tr.t0) - tr.overhead_total)
    push!(tr.bonddims, bond_dims(psi))
    # Handed over by the algorithm, not recomputed: no environment access, no cost.
    push!(tr.energies, real(dE) / tr.nsites)
    push!(tr.errors, eps)
    dt = time() - t_entry
    tr.overhead_total += dt
    push!(tr.overheads, dt)
    return psi, envs
end

"""
    reset!(tracker) -> tracker

Clear the recorded trace and restart the clock.
"""
function reset!(tr::EnergyTracker)
    empty!(tr.iters); empty!(tr.steps); empty!(tr.times); empty!(tr.bonddims)
    empty!(tr.energies); empty!(tr.errors)
    empty!(tr.overheads); tr.overhead_total = 0.0
    tr.t0 = time()
    return tr
end

end # module
