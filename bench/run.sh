#!/usr/bin/env bash
# Sequential benchmark driver: no two runs ever overlap.
set -u
cd "$(dirname "$0")/.."

export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLBACKEND=Agg GKSwstype=100

# run <logname> <julia_threads|-> <cmd...>
run () {
  local name=$1; local jt=$2; shift 2
  local out="bench/$name"
  mkdir -p "$out"
  rm -f energy_vs_time.npy energy_vs_time.png \
        fci_lander_energy_vs_time.npy fci_lander_energy_vs_time.png
  echo "=== $name : start $(date -Is) (JULIA_NUM_THREADS=$jt) ==="
  JULIA_NUM_THREADS="$jt" /usr/bin/time -v "$@" > "$out/stdout.log" 2> "$out/time.log"
  local rc=$?
  echo "=== $name : exit $rc at $(date -Is) ==="
  # snapshot whichever outputs this run produced
  for f in energy_vs_time.npy energy_vs_time.png \
           fci_lander_energy_vs_time.npy fci_lander_energy_vs_time.png; do
    [ -f "$f" ] && mv "$f" "$out/"
  done
  return $rc
}

# Julia run 1 carries JIT/compilation cost; run 2 is the steady-state figure.
run julia_run1     1 julia --project=. script.jl
run julia_run2     1 julia --project=. script.jl
run julia_threads8 8 julia --project=. script.jl
# TeNPy: single process, threaded BLAS.
run python_run1    1 .venv/bin/python script.py
run python_run2    1 .venv/bin/python script.py
echo "ALL DONE $(date -Is)"
