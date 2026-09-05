# Shared Haldane-model definitions: honeycomb lattice geometry, fermionic operators,
# the MPO Hamiltonian, and the charge-shift / ED initial-state helpers.
#
# Extracted verbatim from script.jl so the benchmark script and every probe use one
# definition rather than each carrying its own copy.
#
# Usage:
#     include(joinpath(@__DIR__, "model.jl"))
#     using .HaldaneModel

module HaldaneModel

using LinearAlgebra, MPSKit, TensorKit, MPSKitModels, MatrixAlgebraKit, KrylovKit

export HC_nearest_neighbours, HC_next_nearest_neighbours
export fermion_space, single_site_operator, two_site_operator
export f_num_3, f_plus_f_min_3, myidentity
export HaldaneMPO, shift_my_charge, initialstateED

## ==================================================================
## AnyonKit/src/myhoneycomb.jl
## ==================================================================

"""
Honeycomb lattice geometry for MPSKit.
    Nx = number of unit cells in x direction
    Ny = number of unit cells in y direction

"""

function HC_nearest_neighbours(Nx, Ny, yperiodic = true)
    N = Nx * Ny * 2 # total number of sites (2 per unit cell)
    lat = InfiniteChain(N) # one dimensional infinite lattice with N sites per unit cell
    get_idx(ix, iy, s) = (ix - 1) * (2 * Ny) + (iy - 1) * 2 + 1 + s

    NNbonds = []

    for ix in 1:Nx
        for iy in 1:Ny
            # Site indices for current unit cell
            idxA = get_idx(ix, iy, 0)
            idxB = get_idx(ix, iy, 1)

            # --- Nearest Neighbors ---
            # 1. Intracell bond (A connects to B in same cell)
            push!(NNbonds, (lat[idxA], lat[idxB], false, [sqrt(3) / 2, 1 / 2]))

            # 2. Intercell Y bond: B(x, y) connects to A(x, y+1)
            if iy < Ny
                idxA_nextY = get_idx(ix, iy + 1, 0)
                push!(NNbonds, (lat[idxB], lat[idxA_nextY], false, [sqrt(3) / 2, -1 / 2]))
            elseif yperiodic
                idxA_nextY = get_idx(ix, 1, 0)
                push!(NNbonds, (lat[idxB], lat[idxA_nextY], true, [sqrt(3) / 2, -1 / 2])) # wrap around cylinder
            end

            # 3. Intercell X bond: B(x, y) connects to A(x+1, y)
            # We use ix+1 directly. lat handles indices > N (mapping to next MPS unit cell)
            idxA_nextX = get_idx(ix + 1, iy, 0)
            push!(NNbonds, (lat[idxB], lat[idxA_nextX], false, [0, 1]))
        end
    end
    return NNbonds
end

function HC_next_nearest_neighbours(Nx, Ny, yperiodic = true)
    N = Nx * Ny * 2     # total number of sites (2 per unit cell)
    lat = InfiniteChain(N)
    get_idx(ix, iy, s) = (ix - 1) * (2 * Ny) + (iy - 1) * 2 + 1 + s

    NNNbondsA = []
    NNNbondsB = []

    for ix in 1:Nx
        for iy in 1:Ny
            # --- Next Nearest Neighbors ---
            # 1. Intercell bond Y: (x, y) -> (x, y+1)
            idxA = get_idx(ix, iy, 0)
            idxB = get_idx(ix, iy, 1)
            if iy < Ny
                idxA_nextY = get_idx(ix, iy + 1, 0)
                idxB_nextY = get_idx(ix, iy + 1, 1)
                push!(NNNbondsA, (lat[idxA], lat[idxA_nextY], false, [sqrt(3), 0]))
                push!(NNNbondsB, (lat[idxB], lat[idxB_nextY], false, [sqrt(3), 0]))
            elseif yperiodic
                idxA_nextY = get_idx(ix, 1, 0)
                idxB_nextY = get_idx(ix, 1, 1)
                push!(NNNbondsA, (lat[idxA], lat[idxA_nextY], true, [sqrt(3), 0]))
                push!(NNNbondsB, (lat[idxB], lat[idxB_nextY], true, [sqrt(3), 0]))
            end

            # 2. Intercell bond X: (x, y) -> (x+1, y), doesn't cross PBC
            idxA_nextX = get_idx(ix + 1, iy, 0)
            idxB_nextX = get_idx(ix + 1, iy, 1)
            push!(NNNbondsA, (lat[idxA_nextX], lat[idxA], false, [-sqrt(3) / 2, -3 / 2]))
            push!(NNNbondsB, (lat[idxB_nextX], lat[idxB], false, [-sqrt(3) / 2, -3 / 2]))

            # 3. Intercell bond Diagonal: (x, y) -> (x+1, y-1)
            if iy > 1
                idxA_diag = get_idx(ix + 1, iy - 1, 0)
                idxB_diag = get_idx(ix + 1, iy - 1, 1)
                push!(NNNbondsA, (lat[idxA], lat[idxA_diag], false, [-sqrt(3) / 2, 3 / 2]))
                push!(NNNbondsB, (lat[idxB], lat[idxB_diag], false, [-sqrt(3) / 2, 3 / 2]))
            elseif yperiodic
                idxA_diag = get_idx(ix + 1, Ny, 0)
                idxB_diag = get_idx(ix + 1, Ny, 1)
                push!(NNNbondsA, (lat[idxA], lat[idxA_diag], true, [-sqrt(3) / 2, 3 / 2]))
                push!(NNNbondsB, (lat[idxB], lat[idxB_diag], true, [-sqrt(3) / 2, 3 / 2]))
            end
        end
    end
    return NNNbondsA, NNNbondsB
end

#fermionic operators (e = 3)
fermion_space(::Type{U1Irrep}) = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 3) => 1)
function single_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector} = Trivial)
    V = fermion_space(symmetry)
    return zeros(T, V ← V)
end

function two_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector} = Trivial)
    V = fermion_space(symmetry)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

function f_num_3(T::Type{<:Number}, (::Type{U1Irrep}))
    t = single_site_operator(T, U1Irrep)
    S = sectortype(t)
    # this sets the electron sector to (1,3) i.e. the electron has charge +3.
    block(t, S(1, 3)) .= one(T)
    return t
end

function f_plus_f_min_3(T::Type{<:Number}, ::Type{U1Irrep})
    t = two_site_operator(T, U1Irrep)
    I = sectortype(t)
    # four leg tensor between two sites: [out A, in A, out B, in B]
    t[(I(1, 3), I(0, 0), dual(I(0, 0)), dual(I(1, 3)))] .= 1
    return t
end

function myidentity(T::Type{<:Number}, ::Type{U1Irrep})
    t = single_site_operator(ComplexF64, U1Irrep)
    S = sectortype(t)
    block(t, S(0, 0)) .= 1
    block(t, S(1, 3)) .= 1
    return t
end

# ------- Haldane model Hamiltonian -------------------------------------------------------------

function HaldaneMPO(Nx::Int64, Ny::Int64, t1::Float64, t2::ComplexF64, V1 = t1)
    """
    Hamiltonian for Haldane model with nearest and next-nearest neighbor hopping and zero flux
    """
    hopping = f_plus_f_min_3(ComplexF64, U1Irrep)
    interaction = f_num_3(ComplexF64, U1Irrep) ⊗ f_num_3(ComplexF64, U1Irrep)
    NNbonds = HC_nearest_neighbours(Nx, Ny)
    NNNbondsA, NNNbondsB = HC_next_nearest_neighbours(Nx, Ny)
    # terms and hermitian conjugates
    Hlist = [
        [LocalOperator(hopping, i, j) * (-t1) for (i, j, _, _) in NNbonds]...,
        [LocalOperator(hopping, j, i) * (-t1) for (i, j, _, _) in NNbonds]...,
        [LocalOperator(hopping, i, j) * (t2) for (i, j, _, _) in NNNbondsA]...,
        [LocalOperator(hopping, j, i) * conj(t2) for (i, j, _, _) in NNNbondsA]...,
        [LocalOperator(hopping, i, j) * conj(t2) for (i, j, _, _) in NNNbondsB]...,
        [LocalOperator(hopping, j, i) * (t2) for (i, j, _, _) in NNNbondsB]...,
        [LocalOperator(interaction, i, j) * (V1) for (i, j, _, _) in NNbonds]...,
    ]
    return @mpoham sum(Hlist)
end

# Helper functions
function shift_my_charge(H, Nx, Ny, numel)
    I_sector = FermionParity ⊠ U1Irrep
    if numel == 1   # one electron per 3 unit cells, i.e. 1/3 filling
        Saux = I_sector.(repeat([(1, 1), (0, 0)], Nx * Ny)) # 1/3 filling
    elseif numel == 3 # three electrons per 3 unit cells, i.e. band insulator
        Saux = I_sector.(repeat([(1, 3), (0, 0)], Nx * Ny)) # filled lowest band (band insulator)
    end
    H_shifted = MPSKit.add_physical_charge(H, Saux)
    return H_shifted
end

function initialstateED(H_shifted, Nx, Ny)
    L = Nx * Ny * 2
    Ps = physicalspace(H_shifted)
    Hper = periodic_boundary_conditions(H_shifted, L)
    E, ste = exact_diagonalization(Hper; num = 1, alg = Lanczos(; krylovdim = 200, eager = true))
    AL0 = [ste[1].AL[i] for i in 1:L] # left canonical tensors from middle unit cell
    return InfiniteMPS(AL0)
end

end # module
