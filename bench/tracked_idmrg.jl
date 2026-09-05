# Per-iteration tracking for MPSKit's IDMRG/IDMRG2.
#
# Why this exists rather than a `finalize` callback or an AbstractLogger:
#   * `IDMRG` and `IDMRG2` expose no `finalize` field (verified on MPSKit main);
#     only VUMPS/DMRG/DMRG2/VOMPS/TDVP do.
#   * An AbstractLogger cannot work either: `MPSKit.IterLog` carries only
#     (name, iter, error, objective, t_init, t_prev, t_last, state) and
#     `handle_message` never receives the MPS, so bond dimensions are unreachable.
#
# `find_groundstate_tracked` mirrors `MPSKit._find_groundstate_idmrg` line for line
# (same IterativeSolver, same tol/maxiter checks, same gauge-fixing epilogue) and
# only adds recording, so the algorithm and its cost are unchanged.

module TrackedIDMRG

using MPSKit, TensorKit, TensorOperations
using MPSKit: IDMRG, IDMRG2, IterLog, IterativeSolver, IDMRGState, calc_galerkin,
    default_allocator, recalculate!, adapt_solver, NoTimerOutput, SerialScheduler

export find_groundstate_tracked, IDMRGTrace

"""
Per-iteration record. `bonddims[k]` is the full per-bond dimension profile after
iteration `iters[k]`; `energies` is per site.
"""
struct IDMRGTrace
    iters::Vector{Int}
    times::Vector{Float64}
    energies::Vector{Float64}
    errors::Vector{Float64}
    bonddims::Vector{Vector{Int}}
end
IDMRGTrace() = IDMRGTrace(Int[], Float64[], Float64[], Float64[], Vector{Int}[])

bond_dimensions(mps) = [dim(left_virtualspace(mps, i)) for i in 1:length(mps)]

function find_groundstate_tracked(
        mps, operator, alg::Union{IDMRG, IDMRG2},
        envs = environments(mps, operator, mps);
        nsites::Int = length(mps),
    )
    (length(mps) <= 1 && alg isa IDMRG2) &&
        throw(ArgumentError("unit cell should be >= 2"))

    trace = IDMRGTrace()
    name = alg isa IDMRG ? "IDMRG" : "IDMRG2"
    # Match upstream's verbosity semantics: >3 enables @timeit instrumentation,
    # which is not free, so keep the same choice for cost parity.
    timeroutput = alg.verbosity > 3 ? MPSKit.TimerOutput(name) : NoTimerOutput()

    mps = copy(mps)
    allocator = default_allocator(mps, SerialScheduler())
    ϵ = calc_galerkin(mps, operator, mps, envs; alg.backend, allocator)
    E = expectation_value(mps, operator, envs)

    state = IDMRGState(mps, operator, envs, 0, ϵ, E, timeroutput, allocator)
    it = IterativeSolver(alg, state)

    t0 = time()
    push!(trace.iters, 0)
    push!(trace.times, 0.0)
    push!(trace.energies, real(E) / nsites)
    push!(trace.errors, ϵ)
    push!(trace.bonddims, bond_dimensions(mps))

    # NOTE on the energy: the iterator's 4th value is *named* ΔE upstream, but it is
    # iDMRG's energy-per-unit-cell estimator, `(E_new - E_old) / 2` (plus a factor for
    # 2-site cells) -- not a convergence delta. It is what `logiter!` stores in
    # `IterLog.objective`, so dividing it by the number of sites gives energy per site.
    # `it.state.energy` is the *accumulated total* and is NOT what you want here.
    for (mpsᵢ, _, ϵᵢ, ΔEᵢ) in it
        push!(trace.iters, it.iter)
        push!(trace.times, time() - t0)
        push!(trace.energies, real(ΔEᵢ) / nsites)
        push!(trace.errors, ϵᵢ)
        push!(trace.bonddims, bond_dimensions(mpsᵢ))
        ϵᵢ <= alg.tol && break
        it.iter >= alg.maxiter && break
    end

    alg_gauge = adapt_solver(alg.alg_gauge; iter = it.state.iter, g_global = it.state.ϵ)
    ψ′ = InfiniteMPS(it.state.mps.AR; alg_gauge.tol, alg_gauge.maxiter)
    envs′ = recalculate!(it.state.envs, ψ′, it.state.operator, ψ′)
    return ψ′, envs′, it.state.ϵ, trace
end

end # module
