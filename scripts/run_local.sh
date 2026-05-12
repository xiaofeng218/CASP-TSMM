#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BLAS="${BLAS:-openblas}"
THREADS="${OMP_NUM_THREADS:-$(nproc)}"
BENCH_ARGS=("$@")

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"

echo "=== TSMM local run ==="
echo "BLAS=$BLAS OMP_NUM_THREADS=$OMP_NUM_THREADS"

make BLAS="$BLAS" -j"$(nproc)"
mkdir -p web

python3 web/server.py --port 8080 --results web/results.json &
WEB_PID=$!

cleanup() {
    kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Dashboard: http://localhost:8080"
./benchmark --output web/results.json "${BENCH_ARGS[@]}"
