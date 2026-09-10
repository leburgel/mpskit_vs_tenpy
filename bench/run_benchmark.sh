#!/usr/bin/env bash
# The benchmark: TeNPy then MPSKit. Both set their own BLAS thread count.
# Results and logs land in bench/<name>/.
set -u
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg GKSwstype=100
COOLDOWN=${COOLDOWN:-45}

run () {
  local name=$1; shift
  local out="bench/$name"; mkdir -p "$out"
  local marker="$out/.runstart"; : > "$marker"
  echo "=== $name : start $(date -Is) ==="
  { echo "start: $(date -Is)"
    echo "MPSKit: $(git -C lib/MPSKit.jl log --oneline -1)"
    echo "cmd: $*"; } > "$out/run.log"
  /usr/bin/time -v "$@" > "$out/stdout.log" 2> "$out/time.log"
  echo "=== $name : exit $? at $(date -Is) ==="
  echo "end: $(date -Is)" >> "$out/run.log"
  for f in energy_vs_time.png energy_vs_time.npy fci_lander_energy_vs_time.npy \
           fci_lander_energy_vs_time.png fci_lander_trace.npz; do
    [ -f "$f" ] && [ "$f" -nt "$marker" ] && mv "$f" "$out/"
  done
  rm -f "$marker"
  sleep "$COOLDOWN"
}

run tenpy  .venv/bin/python tenpy_idmrg.py
BENCH_VERBOSITY=3 run mpskit julia --project=. mpskit_idmrg2.jl
echo "DONE $(date -Is)"
