# One importable toolbox: re-exports everything the benchmark scripts need.
#
#     include(joinpath(@__DIR__, "toolbox", "Toolbox.jl")); using .Toolbox
#
# The pieces stay in their own files (model.jl is also included directly by the
# probe scripts), but `Toolbox` re-exports all their public names.

module Toolbox

# The physics stack: used here and re-exported, so scripts need no package imports.
using MPSKit, MPSKitModels, TensorKit, MatrixAlgebraKit, KrylovKit, TensorOperations

include("model.jl")
include("product_start.jl")
include("expansion.jl")
include("energy_tracking.jl")
include("eigsolvers.jl")

using .HaldaneModel
using .ProductStart
using .Expansion
using .EnergyTracking
using .Eigsolvers

# Re-exported programmatically so the list cannot fall out of date.
const SUBMODULES = (HaldaneModel, ProductStart, Expansion, EnergyTracking,
                    Eigsolvers)

# Packages re-exported wholesale, so scripts import only this module.
const REEXPORTED = (MPSKit, MPSKitModels, TensorKit, MatrixAlgebraKit, KrylovKit,
                    TensorOperations)

# On a name clash the first package wins; the clash is recorded in EXPORT_CLASHES
# and the loser stays reachable as `MPSKit.foo` / `TensorKit.foo`.
const EXPORT_CLASHES = Symbol[]

let claimed = Set{Symbol}()
    for M in SUBMODULES, n in names(M)
        n === nameof(M) && continue
        push!(claimed, n)
        @eval export $n
    end
    for M in REEXPORTED, n in names(M)
        n === nameof(M) && continue
        if n in claimed
            n in EXPORT_CLASHES || push!(EXPORT_CLASHES, n)
            continue
        end
        push!(claimed, n)
        @eval export $n
    end
end

# The submodules themselves, for anyone who wants to qualify a name.
export HaldaneModel, ProductStart, Expansion, EnergyTracking, Eigsolvers

# The package *modules* as well as their names, so a script that only does
# `using .Toolbox` can still write `MPSKit.OptimalExpand`, `TensorKit.foo`, or
# reach anything that lost its export to a name clash (see EXPORT_CLASHES).
export MPSKit, MPSKitModels, TensorKit, MatrixAlgebraKit, KrylovKit, TensorOperations
export EXPORT_CLASHES

"""
    initialstate_product(H_shifted, Nx, Ny; occupied, rounds, add, eps, seed)

The benchmark's initial state: a bond-dimension-1 CDW product state, expanded
through `H_shifted` and perturbed until `IDMRG2` can start from it.

The start for every benchmark script. Unlike the ED start it replaced, it is
deterministic, uses ~1.8 GB less memory, converges lower and solves faster.
Rationale in `product_start.jl`, the loop itself in `expansion.jl`.

`rounds = 1` is deliberate: the expansion only has to let `IDMRG2` start, and
IDMRG2 grows the bonds itself from there (2 -> 4 -> 16 -> ... -> 100).

`occupied` are the sites carrying an electron; `[1, 7]` is the CDW pattern used
throughout -- A(y=1) and A(y=4), matching TeNPy's `Cstyle` ordering.
"""
function initialstate_product(
        H_shifted, Nx::Int, Ny::Int;
        occupied = [1, 7], rounds::Int = 1, add::Int = 4,
        eps::Real = 1.0e-6, seed::Int = 1234,
    )
    L = Nx * Ny * 2
    Ps = [physicalspace(H_shifted, i) for i in 1:L]
    # No target, no optimizer: the loop is used purely to expand through H. The
    # perturbation stays outside it, so it draws on `seed` itself.
    psi, = expand_to_target(
        cdw_product_state(Ps, occupied), H_shifted,
        OptimalExpand(; trunc = truncrank(add));
        trunc = notrunc(), add, maxrounds = rounds, verbose = false
    )
    return perturb_state(psi, eps; seed)
end

export initialstate_product

end # module
