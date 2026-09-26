#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export POLARS_MAX_THREADS=4
experiment_python=/home/jy001/micromamba/envs/virtual-cell/bin/python
base_run=("$experiment_python" ../exp003-context-module-cvae/src/runtime.py --base-run)
environment_key=$(pwd | sha256sum | cut -d ' ' -f 1)
exec 9>"${TMPDIR:-/tmp}/vcc-environment-${environment_key}.lock"
flock -x 9
"${base_run[@]}" uv sync --locked --python "$experiment_python" --no-python-downloads
flock -s 9
exec "${base_run[@]}" uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python src/run.py "$@"
