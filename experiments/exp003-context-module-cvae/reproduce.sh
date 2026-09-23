#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
experiment_python=/home/jy001/micromamba/envs/virtual-cell/bin/python
micromamba run -n virtual-cell uv sync --locked --python "$experiment_python" --no-python-downloads
exec micromamba run -n virtual-cell uv run --locked --python "$experiment_python" --no-python-downloads python src/main.py "$@"
