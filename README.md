# Haldane-model iDMRG: MPSKit (Julia) vs TeNPy (Python)

Benchmark comparing MPSKit's `IDMRG2` against TeNPy's `TwoSiteDMRGEngine` on the
Haldane model, width-6 honeycomb cylinder (12 sites/unit cell), `t1 = 1`,
`|t2| = sqrt(129)/36`, `phi = acos(3*sqrt(3/43))`, `V = 1`, 1/3 band filling,
warm-up `chi = 100`.

**Read [FINDINGS.md](FINDINGS.md) first** — it has all measured results, the
structural differences between the two codes, the open questions, and the
methodology warnings you need before trusting any new number.

---

## Setup on a fresh machine

Requires Julia 1.12.2 (via `juliaup`) and Python 3.12.

### 1. Julia

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` pins MPSKit to `main` at commit
`9c9a5c35177f518d5aa2f9ee95636a0ed19b8ecc`, so this is reproducible. To move to
a newer `main`: `Pkg.update("MPSKit")`.

> **MPSKit must be `main`, not a release.** The `IDMRG2` truncation keyword is
> `trunc` on `main` and `trscheme` in released 0.13.13. `script.jl` uses `trunc`.

### 2. Python

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is fully pinned. Check the Cython extensions are active —
without them the comparison is badly skewed:

```bash
.venv/bin/python -c "import tenpy; print(tenpy.tools.optimization.have_cython_functions)"
# must print True
```

### 3. TensorKit `ld-adjoint` branch (optional, not wired in)

Not included in the transfer. Recreate with:

```bash
mkdir -p lib && git clone https://github.com/Jutho/TensorKit.jl.git lib/TensorKit.jl
git -C lib/TensorKit.jl checkout ld-adjoint
```

It is deliberately **not** part of the environment (`Manifest.toml` has zero
path-dependencies). To activate / revert:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="lib/TensorKit.jl")'
julia --project=. -e 'using Pkg; Pkg.free("TensorKit")'
```

See FINDINGS.md section 9.4 for how to A/B it correctly.

---

## Layout

```
model.jl                  module HaldaneModel - the SINGLE definition source
                          (geometry, operators, HaldaneMPO, shift_my_charge,
                          initialstateED). Every script includes this.
script.jl                 original MPSKit benchmark (AbstractLogger tracking)
script_me.jl              variant with a TODO for 1-site VUMPS + bond expansion
script.py                 original TeNPy benchmark

bench/
  tracked_idmrg.jl        find_groundstate_tracked: per-iteration time, energy,
                          Galerkin error, full per-bond chi profile
  cdw_state.jl            deterministic chi=1 CDW product state
  script_tracked.jl       MPSKit: ED start + matched truncation + tracking
  script_tracked.py       TeNPy: order="Cstyle" + ChiTrackingEngine
  script_cstyle.py        minimal order="Cstyle" variant of script.py
  compare.py              tabulate + plot all runs (see caveat below)
  run.sh                  sequential driver (see caveat below)
  probe_*.jl debug_*.jl   diagnostics behind FINDINGS.md sections 5 and 6
  <run-name>/             results: stdout.log, time.log, *.npy/.npz/.png
```

Two caveats carried over from FINDINGS.md:

- `bench/run.sh` runs everything back-to-back, which **contaminates timings** —
  the Python runs came out ~1.8x slow after heavy Julia load. Prefer separate
  runs with cool-down.
- `bench/compare.py`'s *"time to reach threshold"* rows are **invalid**:
  MPSKit's spuriously low iteration-2 energy trips the threshold immediately.
  Its wall/solver/energy/s-per-step columns are fine.

## Running

Run one at a time, matched thread counts:

```bash
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8

JULIA_NUM_THREADS=1 GKSwstype=100 julia --project=. bench/script_tracked.jl
MPLBACKEND=Agg .venv/bin/python bench/script_tracked.py
```

The single most valuable missing measurement is MPSKit at
`OPENBLAS_NUM_THREADS=1`, for a true per-core comparison against TeNPy's
7.82 s/sweep (FINDINGS.md section 9.3).

---

## Transferring this folder

`.venv/` (339 MB) and `lib/` (19 MB) are excluded by `.gitignore` and should not
be copied — both are recreated by the steps above. Everything else is ~750 KB.

```bash
rsync -av --exclude='.venv' --exclude='lib' --exclude='__pycache__' \
      --exclude='*.orig' --exclude='*.bak' \
      ./ user@remote:/path/to/mina/
```

or

```bash
tar --exclude='.venv' --exclude='lib' --exclude='__pycache__' \
    --exclude='*.orig' --exclude='*.bak' -czf mina.tar.gz .
```
