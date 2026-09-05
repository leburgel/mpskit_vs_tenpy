using JLD2, LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit
using Plots, Printf, NPZ

include(joinpath(@__DIR__, "..", "model.jl"));   using .HaldaneModel
include(joinpath(@__DIR__, "cdw_state.jl"));     using .CDWState
include(joinpath(@__DIR__, "tracked_idmrg.jl")); using .TrackedIDMRG
## --- Parameters for Haldane model --

t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1
Nx = 1
Ny = 6

## --- Initialize charge density wave (ED state, exactly as in script.jl) ---
occ = 0
vac = 10
dens = f_num_3(ComplexF64, U1Irrep)
physical_space = space(dens, 1)
chain = fill(physical_space, Ny)
ops = [1 => occ * dens, 2 => vac * dens, 3 => vac * dens, 4 => vac * dens, 5 => vac * dens, 6 => vac * dens, 7 => occ * dens, 8 => vac * dens, 9 => vac * dens, 10 => vac * dens, 11 => vac * dens, 12 => vac * dens]
H_CDW = InfiniteMPOHamiltonian(chain, ops)
H_CDW_shifted = shift_my_charge(H_CDW, Nx, Ny, 1)

t_ed = @elapsed psi1 = initialstateED(H_CDW_shifted, Nx, Ny)
@printf("initialstateED: %.2f s, bond dims = %s\n", t_ed, string(bond_dimensions(psi1)))

## --- build FCI Hamiltonian ---
t_mpo = @elapsed begin
    H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
    H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)
end
@printf("HaldaneMPO build: %.2f s\n", t_mpo)

# DMRG warmup
chi_warmup = 100
tol_warmup = 1.0e-4
maxiter_warmup = 5

# Matches TeNPy's chi_max=100 + svd_min=1e-10 + trunc_cut=1e-14 (the last a default
# that script.py never sets but tenpy applies anyway). The original script used a
# bare truncrank(chi_warmup), i.e. rank only with no singular-value floor.
trunc_matched = truncrank(chi_warmup) &
    trunctol(; rtol = 1.0e-10) &
    truncerror(; atol = 1.0e-14)
println("truncation: ", trunc_matched, "\n")

alg = IDMRG2(;
    trunc = trunc_matched, maxiter = maxiter_warmup, tol = tol_warmup, verbosity = 10
)
t_solve = @elapsed psi1, envs, err, tr = find_groundstate_tracked(
    psi1, H_FCI_shifted, alg; nsites = 12
)
@printf("\nfind_groundstate: %.2f s (final Galerkin error %.3e)\n\n", t_solve, err)

println(rpad("iter", 6), rpad("time [s]", 11), rpad("energy", 16), rpad("error", 12), rpad("max_chi", 9), "bond dims")
println("-"^110)
for k in eachindex(tr.iters)
    @printf(
        "%-6d%-11.3f%-16.8f%-12.3e%-9d%s\n",
        tr.iters[k], tr.times[k], tr.energies[k], tr.errors[k],
        maximum(tr.bonddims[k]), string(tr.bonddims[k])
    )
end

p = plot(
    tr.times, tr.energies; xlabel = "elapsed wall time (s)", ylabel = "Energy",
    marker = :circle, legend = false, title = "IDMRG2 warmup (matched truncation)"
)
savefig(p, joinpath(@__DIR__, "energy_vs_time_matched.png"))
npzwrite(
    joinpath(@__DIR__, "trace_matched.npz"),
    Dict(
        "times" => tr.times, "energies" => tr.energies, "errors" => tr.errors,
        "iters" => tr.iters, "bonddims" => reduce(hcat, tr.bonddims)',
    )
)
println("\nsaved trace_matched.npz")
