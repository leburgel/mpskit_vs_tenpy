# Haldane-model iDMRG: MPSKit (Julia) vs TeNPy (Python)

Same model, same bond dimension, two codes. Haldane model on a width-6 honeycomb
cylinder (12 sites per unit cell) at 1/3 band filling, `chi = 100`.

**Results and conclusions live in [FINDINGS.md](FINDINGS.md).** Short version:
MPSKit is ~3.4x slower per sweep than TeNPy and the two agree on the energy to
2.2e-5. The gap started at ~15x and closed via two changes — MPSKit's MPO was
2.2x wider than TeNPy's, and its default local eigensolver is far more
conservative than this problem needs.

## Layout

```
mpskit_idmrg2.jl          MPSKit benchmark: IDMRG2 from the CDW product state.
                          BENCH_VERBOSITY=3 for clean timings, 4 for the
                          per-stage TimerOutputs split.
tenpy_idmrg.py            TeNPy benchmark: TwoSiteDMRGEngine, order="Cstyle",
                          with per-sweep bond-dimension tracking.

mpskit_vumpssvd_vumps.jl  Variant route: VUMPSSvdCut expansion + VUMPS.
mpskit_optexpand_vumps.jl Variant route: OptimalExpand expansion + VUMPS.
                          Same loop (toolbox/expansion.jl), different expansion
                          primitive. Both are slower and land higher than IDMRG2
                          (FINDINGS section 6); kept as diagnostics of the
                          expansion primitives.

toolbox/
  Toolbox.jl              The one importable entry point; re-exports the rest.
  model.jl                Haldane geometry, operators, HaldaneMPO, shift_my_charge.
  product_start.jl        The initial state: a deterministic chi=1 CDW product
                          state, expanded through H and perturbed so IDMRG2 can
                          start from it.
  expansion.jl            expand_to_target: the (expand -> optimize -> cut) loop
                          both variant routes share. Takes a final truncation
                          (rank or tolerance) and a per-round budget `add`, and
                          translates `add` for either primitive.
  eigsolvers.jl           bench_eigsolve / bench_environments / bench_gauge —
                          the sub-algorithm settings, in one place.
  energy_tracking.jl      EnergyTracker: a `finalize` callback recording
                          per-iteration time, energy, error and bond dimensions.

bench/
  run_benchmark.sh        Runs TeNPy then MPSKit.
  probe_mpodim.jl         Re-check the MPO bond dimension (37 vs TeNPy's 37).
  <run-name>/             Results: stdout.log, time.log, run.log, plus a
                          snapshot of the script and eigsolvers.jl as run.

lib/MPSKit.jl             Dev clone, branch lb/smaller_mpos, with a local
                          `finalize` patch for IDMRG/IDMRG2. Wired in via
                          Project.toml [sources].
```

## Running

Both sides set 8 BLAS threads — the Julia side with `BLAS.set_num_threads`, the
Python side with an `os.environ` preamble ahead of the numpy import (OpenBLAS reads
those variables once, at load time). It is not a controlled variable: BLAS threads
make no difference on this problem (FINDINGS section 5).

```bash
bench/run_benchmark.sh                                    # both codes

MPLBACKEND=Agg .venv/bin/python tenpy_idmrg.py            # TeNPy alone
GKSwstype=100 BENCH_VERBOSITY=3 julia --project=. mpskit_idmrg2.jl   # MPSKit alone
```

**Julia block-buffers stdout when redirected**, so a running job's log stays empty
until it exits. For live output: `script -qec "julia ..." run.log`.

Unit-cell parallelism is available but nearly exhausted (`julia -t 12` gives 1.47x;
FINDINGS section 5).

## Setup on a fresh machine

### Julia

```bash
mkdir -p lib
git clone https://github.com/QuantumKitHub/MPSKit.jl.git lib/MPSKit.jl
git -C lib/MPSKit.jl checkout lb/smaller_mpos    # the narrow-MPO constructor
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` is not tracked, so this resolves fresh; the versions these results
were taken at are in FINDINGS.

`Project.toml`'s `[sources]` points MPSKit at `lib/MPSKit.jl`; TensorKit comes
from the registry (0.17.1). For a path-free environment, replace that entry with
`MPSKit = {rev = "lb/smaller_mpos", url = "https://github.com/QuantumKitHub/MPSKit.jl.git"}`
and re-resolve — but that loses the local `finalize` patch below.

Note the local `finalize` patch in `lib/MPSKit.jl` is uncommitted and required by
`EnergyTracker` — see FINDINGS section 9. `Pkg.free("MPSKit")` is what undoes a
`[sources]` path entry; deleting the entry alone leaves the Manifest dev'd.

### Python

```bash
python3 -m venv --without-pip .venv          # no ensurepip on this box
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import tenpy; print(tenpy.__version__)"
```
