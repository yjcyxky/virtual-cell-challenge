#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
exec micromamba run -n virtual-cell uv run --locked --python python --no-python-downloads python src/main.py "$@"
