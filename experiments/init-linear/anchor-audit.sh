#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BASE_PYTHON=/home/jy001/micromamba/envs/virtual-cell/bin/python
"$BASE_PYTHON" src/anchor_audit.py --gate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
exec micromamba run -n virtual-cell uv run --locked --no-sync --python "$BASE_PYTHON" --no-python-downloads python -u src/anchor_audit.py "$@"
