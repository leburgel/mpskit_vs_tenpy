# ── BLAS threads ────────────────────────────────────────────────────────────
# Must run before numpy/scipy/tenpy are imported: OpenBLAS reads these once, at
# load time. Not a controlled variable -- BLAS threads make no difference here
# (FINDINGS.md section 5). An explicit setting in the environment wins.
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "8")
print(f"BLAS threads: {os.environ['OPENBLAS_NUM_THREADS']} "
      f"({os.cpu_count()} CPUs available)")

TENPY_ORDER = os.environ.get("TENPY_ORDER", "Cstyle")
print(f"MPS site ordering: {TENPY_ORDER}")

import tenpy
import matplotlib.pyplot as plt
import numpy as np
from tenpy.models.haldane import FermionicHaldaneModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

# ── Model parameters ────────────────────────────────────────────────────────
t1 = 1.0
t2_magnitude = np.sqrt(129) / 36
t2_phase     = np.arccos(3 * np.sqrt(3 / 43))
t2           = t2_magnitude * np.exp(1j * t2_phase)
V = 1.0
mu      = 0.0
phi_ext = 0.0

model_params = {
    "Lx"     : 1,
    "Ly"     : 6,
    "bc_y"   : "cylinder",
    "bc_MPS" : "infinite",
    "t1"     : t1,
    "t2"     : t2,
    "V"      : V,
    "mu"     : mu,
    "phi_ext": phi_ext,
    "conserve": "N",
    # Cstyle interleaves A,B per ring, matching Julia's get_idx ordering -- MPO
    # bond dimensions are only comparable between identical orderings.
    # TENPY_ORDER=rings switches to TeNPy's A0..A5 B0..B5 path.
    "order"   : TENPY_ORDER,
}

model = FermionicHaldaneModel(model_params)
lat   = model.lat

# ── Initial CDW product state ───────────────────────────────────────────────
cdw_pattern = ["full", "empty", "empty",   # A sublattice sites 0,1,2
               "empty", "empty", "empty",   # A sublattice sites 3,4,5
               "full", "empty", "empty",   # B sublattice sites 0,1,2
               "empty", "empty", "empty"]   # B sublattice sites 3,4,5

psi0 = MPS.from_product_state(lat.mps_sites(), cdw_pattern, bc=lat.bc_MPS)

# ── DMRG parameters ─────────────────────────────────────────────────────────
CHI_WARMUP = 100
CHI_MAX    = 300
SVD_MIN    = 1e-10

dmrg_params = {
    # Truncation
    "trunc_params": {
        "chi_max" : CHI_MAX,
        "svd_min" : SVD_MIN,
    },
    # Density-matrix mixer: crucial for escaping the CDW initial state. It
    # perturbs the effective Hamiltonian by the reduced density matrix, which is
    # what creates entanglement from scratch.
    "mixer"       : "DensityMatrixMixer",
    "mixer_params": {
        "amplitude"     : 1e-3,   # initial mixing amplitude
        "decay"         : 1.5,    # amplitude decays by this factor each check
        "disable_after" : 15,     # disable the mixer after this many sweeps
    },
    # Convergence criteria
    "max_E_err"     : 1e-6,    # stop when ΔE/site < this between checks
    "max_S_err"     : 1e-5,    # stop when ΔS < this
    "N_sweeps_check": 2,       # check convergence every N sweeps
    "max_sweeps"    : 100,     # hard upper limit on sweeps
    # tenpy >=1.0 asserts max truncation error *after* the run and raises
    # TenpyInconsistencyError. A warm-up from a CDW product state legitimately
    # exceeds it; None downgrades the check to a warning, algorithm unaffected.
    "max_trunc_err" : None,
}

# ── Warm-up sweep at small chi ──────────────────────────────────────────────
print(f"Starting warm-up iDMRG at χ = {CHI_WARMUP} …")

warmup_params = dict(dmrg_params)                     # copy
warmup_params["trunc_params"] = {                     # override chi only
    "chi_max": CHI_WARMUP,
    "svd_min": SVD_MIN,
}
warmup_params["max_sweeps"]   = 20
warmup_params["max_E_err"]    = 1e-4
warmup_params["max_S_err"]    = 1e-3

class ChiTrackingEngine(dmrg.TwoSiteDMRGEngine):
    """Records the full per-bond dimension profile after every sweep.

    ``sweep_stats['max_chi']`` already gives the maximum per convergence check;
    this adds the whole profile so it can be compared bond-by-bond against
    MPSKit's tracked IDMRG2 output.
    """

    def __init__(self, *args, **kwargs):
        # Must exist before super().__init__, which runs environment sweeps.
        self.chi_trace = []
        super().__init__(*args, **kwargs)

    def sweep(self, optimize=True, **kwargs):
        result = super().sweep(optimize=optimize, **kwargs)
        if optimize:  # skip the non-optimizing environment warm-up sweeps
            self.chi_trace.append(list(self.psi.chi))
        return result


eng_warmup = ChiTrackingEngine(psi0, model, warmup_params)
print(f"initial chi: {list(psi0.chi)}")
E_warmup, psi_warmup = eng_warmup.run()
print(f"Warm-up complete.  E/site = {E_warmup:.10f}\n")

# ── Energy vs. elapsed wall time ────────────────────────────────────────────
# sweep_stats['time'] is wall-clock seconds since the engine was created, sampled
# every N_sweeps_check sweeps alongside the energy.
sweeps   = np.array(eng_warmup.sweep_stats["sweep"])
times    = np.array(eng_warmup.sweep_stats["time"])
energies = np.array(eng_warmup.sweep_stats["E"])

max_chis = np.array(eng_warmup.sweep_stats["max_chi"])

print(f"{'sweep':>6} | {'time [s]':>10} | {'E':>18} | {'max_chi':>7}")
print("-" * 51)
for n, t, E, c in zip(sweeps, times, energies, max_chis):
    print(f"{n:6d} | {t:10.2f} | {E:18.10f} | {c:7d}")

print("\nper-bond chi profile after each sweep:")
for k, chis in enumerate(eng_warmup.chi_trace, start=1):
    print(f"  sweep {k:3d}: {chis}")

fig, ax = plt.subplots()
ax.plot(times, energies, marker="o")
ax.set_xlabel("elapsed wall time [s]")
ax.set_ylabel("energy per site")
ax.set_title(f"Warm-up iDMRG convergence (χ = {CHI_WARMUP})")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_path = "fci_lander_energy_vs_time.png"
fig.savefig(plot_path, dpi=150)
print(f"\nSaved plot to '{plot_path}'.")

data_path = "fci_lander_energy_vs_time.npy"
np.save(data_path, np.column_stack([times, energies]))
np.savez(
    "fci_lander_trace.npz",
    times=times, energies=energies, max_chi=max_chis, sweeps=sweeps,
    chi_trace=np.array(eng_warmup.chi_trace),
)
print(f"Saved (time, energy) data to '{data_path}'.")

