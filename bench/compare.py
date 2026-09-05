#!/usr/bin/env python
"""Compare energy-vs-walltime curves from the MPSKit and TeNPy iDMRG runs.

Reads bench/<run>/{energy_vs_time,fci_lander_energy_vs_time}.npy, each an
(n, 2) array of (elapsed seconds, energy per site).
"""
import os
import numpy as np

RUNS = [
    ("julia_run1",     "energy_vs_time.npy",                 "MPSKit (run 1, incl. JIT)"),
    ("julia_run2",     "energy_vs_time.npy",                 "MPSKit (run 2, warm)"),
    ("julia_threads8", "energy_vs_time.npy",                 "MPSKit (8 Julia threads)"),
    ("python_run1",    "fci_lander_energy_vs_time.npy",      "TeNPy (run 1)"),
    ("python_run2",    "fci_lander_energy_vs_time.npy",      "TeNPy (run 2)"),
]
HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    out = []
    for run, fname, label in RUNS:
        p = os.path.join(HERE, run, fname)
        if not os.path.exists(p):
            print(f"[skip] {label}: no {fname}")
            continue
        d = np.load(p)
        out.append((run, label, d[:, 0].astype(float), d[:, 1].astype(float)))
    return out


def wall_from_time_log(run):
    """Total process wall time from /usr/bin/time -v output."""
    p = os.path.join(HERE, run, "time.log")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if "Elapsed (wall clock) time" in line:
            t = line.split()[-1]
            parts = [float(x) for x in t.split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return None


def main():
    data = load()
    if not data:
        print("no data found")
        return

    print(f"{'run':<16} {'points':>6} {'proc wall':>10} {'solver t':>9} "
          f"{'E_final':>16} {'s/step':>8}")
    print("-" * 72)
    for run, label, t, E in data:
        pw = wall_from_time_log(run)
        steps = len(t)
        per = (t[-1] - t[0]) / (steps - 1) if steps > 1 else float("nan")
        print(f"{run:<16} {steps:6d} "
              f"{(f'{pw:.1f}s' if pw else '?'):>10} "
              f"{t[-1]:8.1f}s {E[-1]:16.10f} {per:8.2f}")

    # Energy agreement between the two libraries
    print("\n-- final energy per site --")
    for run, label, t, E in data:
        print(f"  {label:<32} {E[-1]:.10f}")

    # Time to reach a common energy threshold, so unequal sweep counts don't
    # distort the comparison.
    best = max(E.min() for _, _, _, E in data)
    for frac in (1e-2, 1e-3, 1e-4):
        thresh = best + abs(best) * frac
        print(f"\n-- time to reach E <= {thresh:.8f} "
              f"(within {frac:g} of worst-case-best {best:.8f}) --")
        for run, label, t, E in data:
            idx = np.argmax(E <= thresh) if (E <= thresh).any() else None
            if idx is None:
                print(f"  {label:<32} not reached")
            else:
                print(f"  {label:<32} {t[idx]:8.2f}s (step {idx + 1})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    gmin = min(E.min() for _, _, _, E in data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, label, t, E in data:
        ls = "--" if "julia" in run else "-"
        axes[0].plot(t, E, marker="o", ms=3, ls=ls, label=label)
        axes[1].semilogy(t, np.maximum(E - gmin, 1e-16), marker="o", ms=3,
                         ls=ls, label=label)
    axes[0].set_ylabel("energy per site")
    axes[1].set_ylabel("E - global min  (log)")
    for ax in axes:
        ax.set_xlabel("elapsed wall time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("iDMRG warm-up convergence: MPSKit vs TeNPy (Haldane, Ny=6, chi=100)")
    fig.tight_layout()
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
