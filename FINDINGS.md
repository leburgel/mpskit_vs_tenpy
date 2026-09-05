# MPSKit vs TeNPy iDMRG benchmark — findings

Haldane model on a width-6 honeycomb cylinder (12 sites/unit cell), `t1 = 1`,
`|t2| = sqrt(129)/36`, `phi = acos(3*sqrt(3/43))`, `V = 1`, 1/3 band filling
(2 electrons per cell, 1/6 per site), warm-up bond dimension `chi = 100`.

Investigated 2026-09-04. Everything below was measured, not assumed; where a
number is uncertain or confounded it says so explicitly.

---

## 1. TL;DR

- At **matched bond dimension chi=100**, MPSKit is roughly **8x slower per sweep**
  than TeNPy (~10.1 s vs 1.27 s).
- **Most of that gap is threading efficiency, not per-core work.** TeNPy gets
  **6.9x from 8 threads**; MPSKit only reaches ~1.9x on the same 8 BLAS threads.
  Single-threaded TeNPy runs 7.82 s/sweep — the same order as MPSKit's 10.1 s/sweep
  at ~1.9 cores.
- MPSKit uses **~23x more memory** (4.4 GB vs 186 MB).
- TeNPy is **bit-reproducible**; MPSKit is **not** (unseeded `rand!` in
  `exact_diagonalization`), giving ~25% timing and 1.3e-3 energy variance.
- The two scripts were **not** benchmarking the same thing: different initial
  states, different MPS site orderings, different truncation criteria, different
  convergence criteria, and different local eigensolvers.

---

## 2. Environments

| | version | pin |
|---|---|---|
| Julia | 1.12.2 | — |
| MPSKit | `main` @ `9c9a5c35177f518d5aa2f9ee95636a0ed19b8ecc` | `Manifest.toml` |
| TensorKit | 0.17.1 (registry) | `Manifest.toml` |
| MatrixAlgebraKit | 0.6.9 | `Manifest.toml` |
| KrylovKit | 0.10.4 | `Manifest.toml` |
| Python | 3.12.3 | — |
| physics-tenpy | 1.1.1 (Cython extensions **active**) | `requirements.txt` |
| numpy / scipy | 2.5.2 / 1.18.1 (scipy-openblas) | `requirements.txt` |

**MPSKit must be `main`, not a release.** The `IDMRG2` truncation keyword is
`trunc` on `main` and `trscheme` in released 0.13.13. `script.jl` uses `trunc`,
so it only runs against `main`. This is why it appeared "broken" initially.

Hardware for all numbers below: 20 cores, `OPENBLAS_NUM_THREADS=8` unless stated.

---

## 3. Performance results

### 3.1 Sequential suite — **contaminated, do not cite**

`bench/run.sh` ran all five back to back. The Python runs executed immediately
after ~27 min of heavy Julia load and came out **~1.8x slower** than the same
script standalone. Kept for provenance only.

| run | wall | in-solver | final E/site | RSS | CPU |
|---|---|---|---|---|---|
| `julia_run1` | 630.9 s | 191.0 s | -0.2772738984 | 4.56 GB | 213% |
| `julia_run2` | 547.0 s | 136.7 s | -0.2759530639 | 4.42 GB | 191% |
| `julia_threads8` (8 Julia thr) | 447.2 s | 131.9 s | -0.2769777851 | 4.13 GB | 245% |
| `python_run1` | 62.5 s | 56.6 s | -0.2793406487 | 186 MB | 875% |
| `python_run2` | 56.3 s | 50.3 s | -0.2793406487 | 186 MB | 848% |

### 3.2 Standalone runs — **use these**

| run | wall | in-solver | final E/site | RSS | CPU |
|---|---|---|---|---|---|
| TeNPy, 8 threads (x2, identical) | 31.1 s | 27.7 s | -0.2793406487 | 186 MB | ~920% |
| TeNPy, **1 thread** (`bench/python_1thread/`) | 217.3 s | 191.0 s | -0.2793406500 | 186 MB | 99% |
| TeNPy, `Cstyle` order (`bench/tracked_py/`) | — | 29.9 s | -0.2792709199 | 186 MB | — |
| MPSKit, matched truncation (`bench/julia_matched/`) | 235.6 s | 91.5 s | -0.27702758 | 4.36 GB | 186% |

`bench/python_cstyle/` (59.6 s wall / 53.4 s solver) also ran soon after the
suite and is likewise inflated; `bench/tracked_py/` is the clean `Cstyle` number.

### 3.3 The defensible comparison: cost per sweep at chi=100

Only iterations where **both** codes have all bonds at chi=100.

| | s/sweep at chi=100 |
|---|---|
| MPSKit (8 BLAS thr, 186% CPU) | **10.1** (iter 4->5; iter 3->4 was 11.7) |
| TeNPy, 8 threads | **1.27** = (29.94 - 4.53)/20 |
| TeNPy, 1 thread | **7.82** = (190.96 - 34.57)/20 |

TeNPy thread scaling: 190.96 / 27.7 = **6.9x**.

An earlier figure of "12-16x" was wrong — it averaged MPSKit's cheap low-chi
early iterations against TeNPy's all-at-chi=100 sweeps. Use ~8x.

**Not yet measured:** MPSKit at `OPENBLAS_NUM_THREADS=1`. Needed for a true
per-core ratio. This is the single most valuable missing datapoint.

### 3.4 Where MPSKit's wall time actually goes

From `bench/julia_matched/` (total 235.6 s):

| phase | time |
|---|---|
| Julia startup + JIT + plotting | ~83 s |
| `initialstateED` | 53.6 s |
| `HaldaneMPO` build | 7.3 s |
| `find_groundstate` | 91.5 s |

And **within** the 91.5 s solve, only 55.2 s is inside the iteration loop. The
other ~36 s (~40%) is initial environment construction plus the post-loop
epilogue (`InfiniteMPS(it.state.mps.AR)` gauge fixing + `recalculate!` of
environments at chi=100). Invisible to any per-iteration tracking.

Iteration 1 alone took 24.5 s versus 1.5 s for iteration 2 at the same chi=8 —
i.e. ~23 s of JIT inside the first iteration, which the original logger's
timeline attributes to iDMRG work.

---

## 4. Methodology warnings

Any A/B test (e.g. the TensorKit `ld-adjoint` branch) must clear these first.

1. **MPSKit is nondeterministic.** `MPSKit`'s `exact_diagonalization` seeds its
   Lanczos start vector with an unseeded `rand!`
   (`src/algorithms/ED.jl`, `ACs[middle_site] = rand!(similar(ALs[1], ...))`),
   and `script.jl` sets no seed. Consequences measured: final energies differ by
   1.3e-3 between runs (vs `tol_warmup = 1e-4`), in-solver time varies ~25%, and
   `initialstateED` itself ranged 53.6-90.7 s on identical code.
2. **Machine state matters a lot.** Identical TeNPy runs: 27.7 s standalone vs
   50-57 s directly after heavy Julia load. Always cool down / interleave /
   repeat; never trust a single sequential pass.
3. Effects of interest (the TensorKit fix, threading tweaks) are plausibly
   10-30%, i.e. **below the current noise floor**. Seed the RNG and repeat before
   measuring anything.

---

## 5. Structural differences between the two runs

### 5.1 Initial state — they were never the same

| | MPSKit (`initialstateED`) | TeNPy (default `rings`) |
|---|---|---|
| occupied sites | site 1 = A(y=1), site 7 = A(y=4) | `mps[0]` = A(y=0), `mps[6]` = B(y=0) |
| bond dims | `[1,2,4,5,6,7,8,7,6,5,4,2]` | all `1` |
| `<H_FCI>/site` | **0.0000000000** | **+0.1666666667** |
| setup cost | 53.6-90.7 s | ~0 s |
| reproducible | no | yes |

Both have 2 electrons/cell, but MPSKit starts with them far apart on one
sublattice (zero interaction energy) and TeNPy with them on a nearest-neighbour
intracell bond (pure V energy). MPSKit also warns
*"Constructing an MPS from tensors that are not full rank"*.

**Switching TeNPy to `order="Cstyle"` fixes both the ordering and the
occupations at once** — the unchanged `cdw_pattern` then occupies A(y=0) and
A(y=3), which is MPSKit's `[1, 7]`.

### 5.2 MPS site ordering — bond dimensions are otherwise incomparable

| TeNPy `order` | MPS path |
|---|---|
| `default` / `rings` / `snake` | `A0A1A2A3A4A5 B0B1B2B3B4B5` |
| **`Cstyle`** | `A0B0A1B1A2B2A3B3A4B4A5B5` <- matches Julia |
| `snakeCstyle` | `A0B0B1A1A2B2B3A3A4B4B5A5` |

Julia's `get_idx(ix,iy,s) = (iy-1)*2 + 1 + s` interleaves A,B per ring = `Cstyle`.

Ordering is **not** the cause of the performance gap: TeNPy runs at the same
speed either way. It converges to a marginally *worse* energy under `Cstyle`
(-0.2792709 vs -0.2793406), so `rings` is slightly the better representation
here — small (7e-5) but the opposite of the naive expectation.

### 5.3 Truncation criteria

- MPSKit (original): `truncrank(100)` — **rank only, no singular-value floor**.
- TeNPy (effective): `chi_max=100` **and** `svd_min=1e-10` **and**
  `trunc_cut=1e-14` — the last a TeNPy default that `script.py` never sets.

TeNPy normalizes Schmidt values to `sum(S^2)=1`, so `svd_min` is relative.
The matching MPSKit strategy (verified to compose, be accepted by `IDMRG2`, and
reproduce TeNPy's AND-semantics on a synthetic spectrum):

```julia
truncrank(100) & trunctol(; rtol = 1e-10) & truncerror(; atol = 1e-14)
```

Note `trunctol` is keyword-only: `trunctol(; atol, rtol, p, by, keep_below)`.

Empirically this changes **nothing** physically here (final energy lands inside
the existing run-to-run spread), because both codes saturate chi=100 anyway.

### 5.4 Local eigensolver — **not the same algorithm**

`script.py` leaves `diag_method='default'`, and `dmrg.py:734`:

```python
if self.diag_method == 'default':
    max_N = self.options.get('max_N_for_ED', 400, int)
    if self.eff_H.N < max_N:
        E, theta = full_diag_effH(self.eff_H, theta_guess, keep_sector=True)  # dense ED
    else:
        E, theta, N = LanczosGroundState(...).run()
```

**TeNPy uses dense ED whenever the effective Hamiltonian has dimension < 400**,
Lanczos only above that. MPSKit uses Lanczos unconditionally.

| | MPSKit | TeNPy `LanczosGroundState` |
|---|---|---|
| method | Lanczos always | ED if `dim < 400`, else Lanczos |
| max Krylov dim | `krylovdim = 30` | `N_max = 20` (`N_min = 2`) |
| restarts | `maxiter = 200` | none |
| reorthogonalization | `ModifiedGramSchmidt2` (full, twice) | `reortho = False` (none) |
| tolerance | `tol = 1e-10`, adapted by `DynamicTol` -> `clamp(1e-3*eps, 1e-14, 1e-4)` | `P_tol = 1e-14`, adapted -> `max(p_tol_min, min(1e-4, max_trunc_err * 0.05))` |
| energy tol | — | `E_tol = inf` (unused; `E_tol_to_trunc` defaults `None`) |
| cutoff | — | `eps*100 ~ 2.2e-14` |

Both adapt the eigensolver tolerance to the current convergence level. But
MPSKit does **full reorthogonalization against up to 30 Krylov vectors** while
TeNPy does **none against at most 20** — a concrete per-eigensolve cost
difference and an untested optimization lead.

### 5.5 Convergence criteria — measuring different things

**TeNPy** (`dmrg.py:376-400`):

```python
return abs(Delta_E / max(E, 1.0)) < max_E_err and abs(Delta_S) < max_S_err
```

- `E` = iDMRG energy **density**, `(Es[-1] - Es[-delta]) / growth` with
  `growth = age[-1] - age[-delta]`, `delta = min(1 + 2*L, len(age))`. Already
  per-site, which is why TeNPy needs no `/12`.
- `Delta_E`, `Delta_S` are per-sweep changes over `N_sweeps_check`; `S` is the
  mean entanglement entropy over bonds.
- Warm-up sets `max_E_err = 1e-4`, `max_S_err = 1e-3`.
- Convergence does **not** stop the run while the mixer is live
  (`mps_common.py:906-914`): it deactivates the mixer and continues. The warm-up
  ended at sweep 22 by hitting `max_sweeps = 20`, **not** by converging.

**MPSKit**: stops on `eps <= alg.tol` with `eps = norm(C - C_old)` — the change
in the bond matrix, a *state-change* measure, not energy or entropy.

So `tol = 1e-4` means something quite different in each code. MPSKit's final
`eps = 8.488e-2` after 5 iterations (non-monotonic: 1.2e-1, 7.8e-1, 1.7e-1,
3.0e-1, 8.5e-2) says the state is still moving a great deal. **MPSKit was
nowhere near converged; TeNPy essentially was.** Comparing "5 iterations" to
"22 sweeps" was never comparing equal work.

### 5.6 The mixer, and bond-dimension growth

TeNPy's `DensityMatrixMixer` perturbs the reduced density matrices with MPO
terms crossing the centre bond before diagonalizing:

```
rho_L -> tr_R|theta><theta| + a * sum_l h_l tr_R(|theta><theta|) h_l^dagger
```

Mechanically (`_mix_LR`, `mps_common.py:1846`) it reweights the MPO bond index:
every channel gets `amplitude`, except `IdL -> 1` (the true `rho_L`) and
`IdR -> 0`. Then `svd_from_rho` diagonalizes, truncates on `sqrt(val)` after
`val /= sum(val)`, and returns `S = U^dag theta VH^dag` as a **non-diagonal**
bond matrix.

**Call site — after the eigensolve, inside the split:**

```
sweep()                              mps_common.py:394
 -> update_local(theta)              dmrg.py:529
     -> diag(theta)                  1. ED or Lanczos (the actual optimization)
     -> prepare_svd(theta)           2.
     -> mixed_svd(theta)             3. dmrg.py:876  <-- MIXER
         -> mixer.mix_and_decompose_2site -> mix_rho + svd_from_rho
     -> set_B(U, S, VH)              4.
 -> mixer.update_amplitude(sweeps)   after each optimizing sweep, mps_common.py:409
```

Lifecycle: `mixer_activate()` in `pre_run_initialize`, per-sweep amplitude
decay, `mixer_cleanup()` in `post_run_cleanup` (SVDs the 2D `S` back to
diagonal). Applied **per bond, not per sweep**, and only on the side being
updated (`update_LP_RP` -> `mix_left`/`mix_right`).

The mixer does not change the physical state (`theta` is recovered as `U S VH`
up to truncation); it changes **which basis is retained**, admitting Schmidt
states with exactly zero weight in `theta` that are coupled to it by H. The
benefit lands on *subsequent* local updates. For infinite bc with `n=2`,
`get_sweep_schedule` gives `range(0,L) + range(L,0,-1)` = **24 bond updates per
sweep** for L=12, each applying the mixer — hence the very fast compounding.

Schedule in `script.py`: amplitude `1e-3`, `/1.5` **per sweep** (the script
comment says "each check", which is wrong — with `N_sweeps_check=2` the decay is
twice as fast as implied), disabled at sweep 15, by which point the amplitude is
`1e-3/1.5^15 ~ 2.3e-6`.

**Measured chi growth:**

| after | MPSKit `IDMRG2` | TeNPy (mixer) |
|---|---|---|
| start | `[1,2,4,5,6,7,8,7,6,5,4,2]` | all 1 |
| 1 sweep | max 8 | `[8,4,100,8,11,16,19,23,31,33,30,16]` |
| 2 | max 32 | **all 100** |
| 3 | max 100 (not all bonds) | all 100 |
| 4 | **all 100** | all 100 |

`IDMRG2` has no mixer; its split is a plain `svd_trunc!` of the optimized
two-site tensor, so it can only retain sectors already present in `theta`.

---

## 6. Why `IDMRG2` fails from a chi=1 product state

Reproducer: `bench/debug_lapack.jl`, `bench/debug_sweep.jl`, `bench/debug_svd.jl`.

`find_groundstate(product_state, H, IDMRG2(...))` dies with
`LAPACKException(11)`. **It is not the SVD** — the stacktrace is:

```
stegr!                       linalg.jl:44/447   <-- LAPACK symmetric tridiagonal eigensolver
tridiageigh!                 linalg.jl:114
#eigsolve#54                 lanczos.jl:59      <-- KrylovKit Lanczos
fixedpoint                   fixedpoint.jl:16
_localupdate_sweep_idmrg2!   idmrg.jl:240/243   <-- the two-site local update
iterate                      idmrg.jl:144/145
```

Two independent contributing causes, both measured:

**(a) Charge locking makes the two-site space 1-dimensional.** With chi=1 on
both flanking bonds the left virtual space carries a definite charge `a` and the
right a definite `r`, so fermion-parity x U(1) conservation forces the two
physical indices to total exactly `r - a`. Measured at pos 2: codomain fuses to
`{(0,2), (1,5)}`, domain to `{(0,2), (1,-1)}`, intersection is the single sector
`(0,2)` with block size `(1,1)`.

```
pos 1:  2 coupled sectors, svd [0.5257], [0.8507]  -> bond can reach 2
pos 2:  1 coupled sector,  svd [1.0]               -> bond pinned at 1
pos 3:  1 coupled sector,  svd [1.0]               -> bond pinned at 1
```

There is nothing to optimize and nothing to entangle. The whole left-to-right
leg completes with chi staying 1. At 1/6 filling almost every two-site block is
(empty, empty), which is uniquely determined.

**(b) MPSKit runs Lanczos on those 1x1 / 2x2 blocks** with `krylovdim = 30`,
where TeNPy would dispatch to dense ED (`dim < 400`). `stegr!` is handed a
degenerate tridiagonal problem built from a Krylov space exhausted (or exactly
zero — measured `norm(H*ac2) = 0` at pos 2 and 3) after one step.

Things tried that do **not** fix it:
- bare `truncrank(20)`, `& trunctol`, `& truncerror` — all fail identically, so
  the truncation strategy is not implicated;
- `changebonds(psi, RandExpand(; trunc=truncrank(k)))` — preserves the state
  correctly (occupations and energy unchanged) but only reaches chi=2 on 4 of 12
  bonds regardless of target `k`, and the failure persists.

The `bench/cdw_state.jl` product state itself is verified correct: bond dims all
1, `<n_i>` exactly on the requested sites, `norm = 1`, unit cell closes at net
charge zero, `<H>/site = 0.0` matching the ED state.

**Probably worth filing upstream:** no dense-ED fallback for tiny effective
Hamiltonians, and/or KrylovKit not handling an exhausted Krylov space. A minimal
reproducer is `bench/debug_lapack.jl`.

---

## 7. Correctness notes and small bugs

1. **`script.jl`'s `/12` is correct.** The IDMRG iterator yields
   `(mps, envs, eps, dE)` where `dE` is *named* a delta upstream but is the
   **energy-per-unit-cell estimator**:
   ```julia
   dE = (E_new - state.energy) / 2
   (alg_type <: IDMRG2 && length(mps) == 2) && (dE /= 2)  # "correct energy per unit cell"
   ```
   That is what `logiter!` stores in `IterLog.objective`. **`it.state.energy` is
   the accumulated total and must NOT be used** — doing so produces nonsense
   (-0.656, -1.174 instead of -0.307, -0.262).
2. **`max_trunc_err` blocks `script.py` on tenpy >= 1.0.** A post-run assertion
   (`mps_common.py:810`, default 1e-4) raises `TenpyInconsistencyError` *after*
   the DMRG completes, killing the script before it can plot/save. Warm-up from
   a CDW product state legitimately exceeds it. Fixed by `"max_trunc_err": None`
   in `dmrg_params` (downgrades to a warning; the option is read only by that one
   check, so the algorithm is unaffected).
3. **TeNPy docstring/code mismatch:** `is_converged`'s docstring says
   `|Delta S|/S < max_S_err` (relative); the code does `abs(Delta_S) < max_S_err`
   (absolute).
4. **`script.py` mixer comment is wrong:** "decays by this factor each check" —
   the decay is per *sweep* (`mps_common.py:407-409`), so with `N_sweeps_check=2`
   it is twice as fast as the comment implies.
5. **No `finalize` on `IDMRG`/`IDMRG2`** in released MPSKit *or* `main` — only
   `VUMPS`, `DMRG`, `DMRG2`, `VOMPS`, `TDVP`, `BUG` have one. An `AbstractLogger`
   cannot substitute for tracking bond dimensions either: `IterLog` carries only
   `(name, iter, error, objective, t_init, t_prev, t_last, state)` and
   `handle_message` never sees the MPS. Hence `bench/tracked_idmrg.jl`.
6. **`verbosity = 10` costs real time.** Upstream enables `@timeit`
   instrumentation when `verbosity > 3`. Untested how much.

---

## 8. Infrastructure built

| file | purpose | status |
|---|---|---|
| `model.jl` | module `HaldaneModel`: geometry, operators, `HaldaneMPO`, `shift_my_charge`, `initialstateED`. **Single definition source** — every script includes it. | working |
| `bench/tracked_idmrg.jl` | `find_groundstate_tracked` — per-iteration time / energy / Galerkin error / **full per-bond chi profile**. Mirrors `_find_groundstate_idmrg` line-for-line (same `IterativeSolver`, tol/maxiter checks, gauge-fixing epilogue), adds only recording. | validated: reproduces the logger's -0.30671 / -0.26181 vs -0.30669 / -0.26180 |
| `bench/cdw_state.jl` | deterministic bond-dimension-1 CDW product state + `occupation_operator`, `bond_dimensions`. | validated, but `IDMRG2` cannot start from it (section 6) |
| `bench/script_tracked.jl` | MPSKit run: ED start, matched truncation, tracked driver, saves `trace_matched.npz`. | working |
| `bench/script_tracked.py` | TeNPy run: `order="Cstyle"` + `ChiTrackingEngine` recording the full chi profile per sweep, saves `fci_lander_trace.npz`. | validated behaviourally identical to uninstrumented (same energy to 10 digits) |
| `bench/script_cstyle.py` | minimal `order="Cstyle"` variant of `script.py`. | working |
| `bench/compare.py` | reads all `bench/*/‌*.npy`, tabulates wall/solver/energy/s-per-step, plots `comparison.png`. | working; **its "time to reach threshold" rows are invalid** — MPSKit's spuriously low iteration-2 energy trips the threshold immediately |
| `bench/run.sh` | sequential driver, snapshots each run's outputs into `bench/<name>/`. | works, but see section 4.2 — sequential ordering contaminates results |
| `bench/probe_*.jl`, `bench/debug_*.jl` | the diagnostics behind sections 5 and 6. | working |

Backups of the pre-refactor originals: `script.jl.orig`, `script_me.jl.bak`.

---

## 9. Next steps

**Prerequisites before any A/B measurement**

1. Seed the RNG (`Random.seed!` before `initialstateED`) or bypass ED entirely.
   Without this the noise floor exceeds the effects of interest.
2. Repeat runs with cool-down; never a single sequential pass.

**The measurement that's missing**

3. MPSKit at `OPENBLAS_NUM_THREADS=1` for a true per-core ratio against TeNPy's
   7.82 s/sweep. This is the highest-value single datapoint.

**Prepared but not started**

4. `lib/TensorKit.jl` is cloned at branch `ld-adjoint` and deliberately **not**
   wired into the environment (`Manifest.toml` has zero path-deps). Activate with
   `Pkg.develop(path="lib/TensorKit.jl")`, revert with `Pkg.free("TensorKit")`.
   The branch is **1 commit ahead of and 1 behind `origin/main`**, so A/B it
   against `origin/main`, not registry 0.17.1, or the refactor commit and branch
   drift get mixed into the measurement. Relevant commits:
   ```
   02a2fbe Refactor index manipulation kernels around position-indexed subblocks
   b909280 perf: fix performance regression for non-abelian index manipulations (#521)
   c7169c0 Add `hash` for `FusionTreeBlock` to fix cacheing behind `AdjointTensorMap` `permute`s (#518)
   ```
   Caveat: the sector here is `fZ2 x U1Irrep`, which is **abelian**, so #521
   should not move this benchmark. #518 (caching behind `AdjointTensorMap`
   permutes) is the plausible one, since iDMRG permutes adjoints constantly.

**Untested optimization leads**

5. `verbosity <= 3` to drop TimerOutputs instrumentation.
6. MPSKit's full `ModifiedGramSchmidt2` reorthogonalization (krylovdim 30) vs
   TeNPy's none (N_max 20).
7. The ~36 s of the 91 s solve spent outside the iteration loop (environment
   setup + gauge-fixing epilogue).
8. Threading: MPSKit reaches only ~1.9x on 8 BLAS threads vs TeNPy's 6.9x. Given
   section 3.3 this is where most of the gap lives.

**Open question**

9. Genuinely identical initial states remain blocked by section 6. Options:
   give MPSKit a mixer-equivalent, add a dense-ED fallback for tiny blocks, or
   accept documented-different starts and compare only at matched chi.
