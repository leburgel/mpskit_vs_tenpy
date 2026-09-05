using JLD2, LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit
using Dates, TOML
using Plots
using TensorOperations
using Logging, Printf, NPZ

# Lattice geometry, fermionic operators, HaldaneMPO, shift_my_charge, initialstateED
include(joinpath(@__DIR__, "model.jl"))
using .HaldaneModel

## --- Parameters for Haldane model --

t1 = 1.0
phiphase = acos(3 * sqrt(3 / 43))
t2 = 1 / 12 * sqrt(43 / 3) * exp(im * phiphase)
V1 = 1
Nx = 1
Ny = 6 # circumference of cylinder (number of unit cells in y direction)
## --- Initialize charge density wave ---

occ = 0
vac = 10
dens = f_num_3(ComplexF64, U1Irrep)
physical_space = space(dens, 1)
chain = fill(physical_space, Ny)
ops = [1 => occ * dens, 2 => vac * dens, 3 => vac * dens, 4 => vac * dens, 5 => vac * dens, 6 => vac * dens, 7 => occ * dens, 8 => vac * dens, 9 => vac * dens, 10 => vac * dens, 11 => vac * dens, 12 => vac * dens]
H_CDW = InfiniteMPOHamiltonian(chain, ops)
H_CDW_shifted = shift_my_charge(H_CDW, Nx, Ny, 1) # shifted to 1/3 filling
psi1 = initialstateED(H_CDW_shifted, Nx, Ny) # initial state

## --- build FCI Hamiltonian ---
H_FCI = HaldaneMPO(Nx, Ny, t1, t2, V1)
H_FCI_shifted = shift_my_charge(H_FCI, Nx, Ny, 1)

# DMRG warmup
chi_warmup = 100
tol_warmup = 1.0e-4
maxiter_warmup = 5
verbosity_warmup = 3

mutable struct EnergyTrackingLogger <: AbstractLogger
    inner::AbstractLogger
    times::Vector{Float64}
    energies::Vector{Float64}
end
EnergyTrackingLogger(inner) = EnergyTrackingLogger(inner, Float64[], Float64[])

Logging.min_enabled_level(l::EnergyTrackingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::EnergyTrackingLogger, args...) = Logging.shouldlog(l.inner, args...)
Logging.catch_exceptions(l::EnergyTrackingLogger) = Logging.catch_exceptions(l.inner)

function Logging.handle_message(l::EnergyTrackingLogger, level, message, _module, group, id, file, line; kwargs...)
    if message isa MPSKit.IterLog && message.objective !== nothing
        push!(l.times, message.t_last - message.t_init)
        push!(l.energies, real(message.objective) / 12) # divide by number of sites to get energy per site
    end
    return Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
end

tracker = EnergyTrackingLogger(current_logger())

alg = IDMRG2(;
    trunc = truncrank(chi_warmup),
    maxiter = maxiter_warmup,
    tol = tol_warmup,
    verbosity = verbosity_warmup
)

psi1, envs, err = with_logger(tracker) do
    find_groundstate(psi1, H_FCI_shifted, alg)
end
times, energies = tracker.times, tracker.energies

println(rpad("iter", 6), rpad("time [s]", 12), "energy")
println("-"^40)
for (i, (t, E)) in enumerate(zip(times, energies))
    @printf("%-6d%-12.3f%.8f\n", i, t, E)
end

p = plot(
    times, energies; xlabel = "elapsed wall time (s)", ylabel = "Energy",
    marker = :circle, legend = false, title = "IDMRG2 warmup convergence"
)
savefig(p, joinpath(@__DIR__, "energy_vs_time.png"))

npzwrite(joinpath(@__DIR__, "energy_vs_time.npy"), hcat(times, energies))
