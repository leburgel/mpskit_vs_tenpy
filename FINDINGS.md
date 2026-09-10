# MPSKit vs TeNPy iDMRG — findings

Haldane model on a width-6 honeycomb cylinder (12 sites per unit cell), `t1 = 1`,
`|t2| = sqrt(129)/36`, `phi = acos(3*sqrt(3/43))`, `V = 1`, 1/3 band filling
(2 electrons per cell), `chi = 100`. Machine: 72-core dual-socket Xeon Gold 6140,
376 GB; Julia 1.12.7, Python 3.12.3.

Validated conclusions only; the 2026-09-04..08 chronology is dropped.

---

## 1. Result

The shipped configuration, `chi = 100`, steady state, 8 BLAS threads, no solver
instrumentation (`BENCH_VERBOSITY=3`). Both measured 2026-09-09:
`bench/twopass_tenpy`, `bench/twopass_idmrg2`.

| | s per sweep/iteration | E/site | peak RSS |
|---|---|---|---|
| TeNPy, `TwoSiteDMRGEngine` | **1.233** | -0.2792287344 | 0.19 GB |
| MPSKit, `IDMRG2` | **4.21** | **-0.2792506423** | 3.87 GB |

**MPSKit is ~3.4x slower per sweep and uses ~20x the memory; the two agree on the
energy to 2.2e-5, MPSKit marginally lower.**

The gap was ~15x at the start. Two changes closed it, each worth roughly 1.8x, and
both *improved* accuracy:

| step | s/iteration |
|---|---|
| starting point (82-wide MPO, stock solver) | 26.7 |
| MPO bond dimension 82 -> 37 (section 2) | 15.0 |
| local solver settings (section 3) | 7.44 |
| same settings, timers off, 8 BLAS threads | **4.21** |

The first three rows are a like-for-like A/B measured at `verbosity = 4` on one
BLAS thread, so their absolute values carry the instrumentation; the last row is
what the repo actually runs. A TeNPy sweep and an `IDMRG2` iteration are
like-for-like: both do 24 two-site updates over the 12-site cell.

---

## 2. Cause 1: the MPO was 2.2x too wide

MPSKit's `HaldaneMPO` had bond dimension **82 on every bond**, TeNPy's **37** for
the like-for-like `Cstyle` ordering (32-40 across its orderings). Two-site
application cost is linear in this.

Fixed upstream by `lb/smaller_mpos`, commit `809c8109` ("De-duplicate channels
where possible in `MPOHamiltonian` constructor"), one commit on `main` (`4b579944`).

| | before | after | TeNPy `Cstyle` |
|---|---|---|---|
| Ny=6 (12 sites), per bond | 82 | **37** | 37 interior, 32 at cell edges |
| Ny=3 (6 sites), per bond | 49 | **22** | — |

**Validated by bit-identical energies** at every one of 50 iterations
(-0.2792370806 variational, -0.2792627161 estimator): the same Hamiltonian with the
redundancy removed. Cost 1255.1 -> 725.7 s (1.73x), peak RSS 6.05 -> 4.78 GB.

`shift_my_charge` does not affect the width, and it is not an ordering artifact on
either side. Uninvestigated: MPSKit's 37 is flat across bonds while every TeNPy
ordering tapers to 32 at the cell edges.

---

## 3. Cause 2: local solver settings

Stock `Defaults.alg_eigsolve()` is far more conservative than this problem needs.
The final configuration, on the 37-wide MPO:

```julia
using MPSKit, TensorKit   # truncrank comes from MatrixAlgebraKit via TensorKit

alg = IDMRG2(;
    trunc = truncrank(100), maxiter = 50, tol = 1.0e-4, verbosity = 3,
    alg_eigsolve = MPSKit.Defaults.alg_eigsolve(;
        krylovdim = 16, maxiter = 5, eager = true, ishermitian = true,
        tol_factor = 1.0e0,
        tol_max = 1.0e-2,      # <-- the knob that matters
    ),
)
```

`alg_gauge` stays at its default, and so does KrylovKit's orthogonalizer.

One BLAS thread, `verbosity = 4` throughout, so the rows are comparable to each
other rather than to section 1:

| config | s/iteration | E/site variational | Galerkin err |
|---|---|---|---|
| stock (`krylovdim` 30, `maxiter` 200, `tol_max` 1e-4, `tol_factor` 1e-3) | 13.62 | -0.2792370806 | 8.877e-3 |
| `krylovdim` 16, `maxiter` 5 | 8.12 | -0.2792383005 | 8.877e-3 |
| + `tol_max` 1e-2 | **7.44** | **-0.2792506423** | **1.412e-3** |

**1.8x faster than stock, with a lower energy and a 6x smaller Galerkin error.**

- **`tol_max` is the binding knob, not `tol_factor`.** The dynamic tolerance is
  `clamp(tol_factor * g_global / sqrt(iter), tol_min, tol_max)`. With the Galerkin
  error at ~8e-3 any `tol_factor >= 1e-1` saturates against the ceiling, so raising
  the ceiling is the only way to go looser.
- **Looser local solves converge the outer iteration better** (6x smaller error).
  Likely the `/sqrt(iter)` damping: a tight ceiling forces accuracy into early
  iterations, where the state does not deserve it.

### 3.1 For VUMPS the knob is `alg_environments`

A VUMPS iteration is ~50% environment recomputation, which runs through
`alg_environments` — a separate algorithm with its own `DynamicTol`. Relaxing
`alg_eigsolve` therefore does almost nothing for VUMPS; relaxing
`alg_environments` the same way gives **10.23 -> 6.74 s/iteration** (1.52x), again
with energy and error slightly improved.

Only `VUMPS` exposes the field. `IDMRG`/`IDMRG2` transfer environments
incrementally (10% of an iteration), and `VUMPSSvdCut` calls bare `environments`
internally, so its rebuilds are unreachable this way (section 6).

---

## 4. Where the time goes

MPSKit's own `TimerOutputs`, live only at `verbosity > 3`. `IDMRG2`, final
configuration, 50 iterations / 253 s:

```
 Section          ncalls    time    %tot      GC      avg
 localupdate          50    232s  100.0%   77.6s   4.64s
 ├─ AC2_eigsolve   1.20k    193s   83.3%   72.4s   161ms
 ├─ transfer_env   1.20k   23.4s   10.1%   5.07s  19.5ms
 └─ svd_trunc      1.20k   3.96s    1.7%   156ms  3.30ms
 finalize             50    570μs    0.0%       ∅  11.4μs
```

- **`AC2_eigsolve` is 83% of the run.** Truncation and environment transfers are
  not worth optimising.
- **Allocation is the largest remaining inefficiency, and it is untouched.** A
  50-iteration run moves ~400 GiB with **~30% of runtime in GC** (77.6 s of 232 s).
  Buffer reuse in the two-site application could recover roughly a quarter of the
  runtime. Nobody has looked at this.
- Our own instrumentation is free: `EnergyTracker` costs 570 μs per 279 s solve.

---

## 5. Threading

**BLAS threads make little difference for either code**, so BLAS threading is not
a controlled variable here: every script just sets 8 and moves on. Measured
1 -> 8 threads on the shipped configuration: `IDMRG2` 4.64 -> 4.21 s/iteration
(1.10x, `bench/blas1_idmrg2` vs `bench/twopass_idmrg2`), TeNPy 1.486 -> 1.233
s/sweep (1.21x, `bench/abs_tenpy` vs `bench/twopass_tenpy`). TeNPy was separately
measured flat from 1 to 72 threads while CPU consumption grew 35x.

**Unit-cell parallelism gives 1.47x on 12 threads (~12% efficiency), and is
exhausted.** MPSKit selects a `DynamicScheduler` whenever `Threads.nthreads() > 1`.
VUMPS, 12 threads on the 12-site cell, BLAS pinned to 1:

| | s/VUMPS-iteration | peak RSS |
|---|---|---|
| 1 thread | 6.74 | 5.4 GB |
| 12 threads | **4.59** | 8.6 GB |

Energies are bit-identical, so threading is numerically inert. The stage split
explains the low efficiency: `localupdate` parallelises nearly perfectly (share
45.7% -> 5.4%), but **`envs` does not** — `left_envs` and `right_envs` are each
~186 s inside a 190 s parent, so they run concurrently with each other and each is
internally serial, and its share *rises* to 82%. The environment fixed-point sweep
is sequential along the unit cell by construction; going further needs a different
algorithm, not a scheduler setting.

---

## 6. Warm-up route: `IDMRG2` wins

Three routes from the same deterministic CDW product start to `chi = 100`, all
reaching every bond, all with the final solver settings:

| route | script | E/site | total solve |
|---|---|---|---|
| `IDMRG2` | `mpskit_idmrg2.jl` | **-0.2792506423** | 245 s |
| `VUMPSSvdCut` expansion + VUMPS | `mpskit_vumpssvd_vumps.jl` | -0.2791436977 | 595 s |
| `OptimalExpand` expansion + VUMPS | `mpskit_optexpand_vumps.jl` | -0.2788578016 | 581 s |

The energies are directly comparable; the totals are not measured under identical
conditions (the VUMPS rows carry live timers), but the factor-of-two gap is far
larger than that difference.

`IDMRG2` reaches the lowest energy at less than half the cost: its two-site update
grows the bonds *and* optimises in the same step, so nothing is paid at full `chi`
before the state deserves it. The expansion routes are diagnostics of the expansion
primitives, not competitive warm-ups; both drive the same loop
(`expand_to_target`, `toolbox/expansion.jl`).

- **`VUMPSSvdCut` rebuilds every environment once per site.** `changebonds_n`
  re-gauges the whole MPS and calls `environments(state, H, state)` for each site,
  discarding each rebuild on the next iteration, for a change that touched two
  sites. This dominates that warm-up. The fix is the incremental
  `transfer_leftenv!`/`transfer_rightenv!` `IDMRG2` already uses, or hoisting the
  rebuild out of the per-site loop. Worth reporting upstream. **The user deferred
  fixing it; do not start unprompted.**
- **`OptimalExpand` is state-preserving**, so expanding to `chi = 100` without
  optimising in between leaves the state energetically *still the product state*
  (E/site = +1.3e-6 after 14 rounds), and VUMPS then pays full `chi` cost from
  scratch. Interleaving VUMPS iterations fixes this only once `alg_environments` is
  relaxed (section 3.1): stock gives E/site -0.2492 and error 2.7e-1, relaxed gives
  -0.2789. Two earlier explanations for that divergence — near-singular bond
  matrices, then rank deficiency of the padded bonds — were tested and are **wrong**.

---

## 7. Correctness and equivalence

- **The codes agree.** Converged at `chi = 100`: TeNPy -0.2792287344, MPSKit
  -0.2792506423. Every larger discrepancy in earlier work was non-convergence, not
  physics: `IDMRG2` stopped at 5 iterations gives -0.2765905800, still climbing.
- **Convergence criteria are not comparable, and this matters.** MPSKit stops on
  `norm(C - C_old)`, TeNPy on per-sweep changes in energy density and entanglement
  entropy, so `tol = 1e-4` means different things — MPSKit's error reaches only
  1.4e-3 after 50 iterations while its energy is within 1.6e-4 of final by
  iteration ~10. **Compare at matched iteration counts and energies, never by
  trusting `tol`.**
- **`IDMRG2`'s per-iteration figure is not a variational bound.** It is `ΔE` per
  unit cell (upstream's `IterLog.objective`), the same class of estimator as
  TeNPy's `sweep_stats['E']`. It tracks the variational `expectation_value` to
  2.6e-5 at `chi = 100` but can sit *below* the true ground state at small `chi`
  (-0.28294 at `chi = 4`). `mpskit_idmrg2.jl` prints both.
- **Determinism.** The CDW product start (`initialstate_product`) is
  bit-reproducible across runs. It replaced an `exact_diagonalization` start whose
  unseeded `rand!` put a ~30% spread on s/sweep; do not go seeding RNGs, that path
  is gone. Wall clock still varies ~11% run to run, so repeat *timing* comparisons.

---

## 8. Reproducing this

```bash
.venv/bin/python tenpy_idmrg.py
# BENCH_VERBOSITY=3 for clean timings, 4 for the stage split
BENCH_VERBOSITY=3 julia --project=. mpskit_idmrg2.jl
```

Solver settings live in one place, `toolbox/eigsolvers.jl`: `bench_eigsolve()`,
`bench_environments()`, `bench_gauge()`. Every script uses them.

`bench/<run>/` keeps the runs backing the numbers above, each with `stdout.log`,
`time.log` and a snapshot of the script and `eigsolvers.jl` as run. Superseded run
directories were deleted, so **this document is the primary record for them**: a
number in a table above can be re-measured, but not re-checked against a log.

**Environment gotchas.**

- **Julia block-buffers stdout when it is redirected**, so a running job's log
  stays empty until it exits — indistinguishable from a hang. Use
  `script -qec "julia ..." run.log` for live output.
- **Don't gate a wait loop on `pgrep -f <pattern>` that your own process carries.**
  A driver polling for `julia --project=. <script>` matched its own launching
  shell, whose command line contained the script text, and waited on itself.
- Editing anything under `lib/MPSKit.jl/src/` invalidates the precompile cache; the
  next `julia` start pays ~2 min. Budget for it in timeouts.
- No `ensurepip` on this box: `python3 -m venv --without-pip .venv`, then bootstrap
  with `get-pip.py`.
- `Project.toml`'s `[sources]` points MPSKit at `lib/MPSKit.jl`, which must be on
  `lb/smaller_mpos` with the `finalize` patch applied. A path-free environment
  (`{rev = "lb/smaller_mpos", url = ...}`) costs that patch, and hence
  `EnergyTracker`. `Pkg.free` is what undoes a `[sources]` path; removing the entry
  alone leaves the Manifest dev'd.

---

## 9. MPSKit state: local patches

`lib/MPSKit.jl` is on `lb/smaller_mpos` (`809c8109`) with two uncommitted changes:

1. **`finalize` for `IDMRG`/`IDMRG2`** (`src/algorithms/groundstate/idmrg.jl`).
   Upstream has no such hook for the infinite algorithms, which is why
   `EnergyTracker` needs it. **It diverges from upstream on purpose**:
   `finalize(iter, ψ, H, envs, ϵ, ΔE)`, six arguments, so the callback receives the
   algorithm's own energy and error. The file carries a `LOCAL DIVERGENCE FROM
   UPSTREAM — REVERT BEFORE ANY UPSTREAM PR` block with revert instructions.
2. A **comment only** in `changebonds/vumpssvd.jl`, flagging the per-site
   environment rebuild of section 6.

`lib/TensorKit.jl` no longer exists: the `ld-adjoint` branch made no measurable
difference, so TensorKit is back to registry **0.17.1** (also the newest tag); the
branch survives at `origin/ld-adjoint`.
