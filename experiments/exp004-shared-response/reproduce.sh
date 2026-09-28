#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
experiment_python=/home/jy001/micromamba/envs/virtual-cell/bin/python
runtime=../exp003-context-module-cvae/src/runtime.py
base_run=("$experiment_python" "$runtime" --base-run)
entrypoint=src/run.py
if [[ "${1:-}" == "--export-checkpoint" ]]; then
  entrypoint=src/export.py
  shift
fi
environment_directory=$(pwd)
if [[ -d .venv ]]; then
  environment_directory=$(dirname "$(readlink -f .venv)")
fi
environment_key=$(printf '%s\n' "$environment_directory" | sha256sum | cut -d ' ' -f 1)
exec 9>"${TMPDIR:-/tmp}/vcc-environment-${environment_key}.lock"
flock -s 9
if ! { test -x .venv/bin/python && "${base_run[@]}" uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python "$runtime" --experiment .; }; then
  if [[ "$entrypoint" == src/export.py ]]; then
    echo 'Checkpoint export requires the existing verified environment; no synchronization is attempted.' >&2
    exit 1
  fi
  flock -u 9
  flock -x 9
  "${base_run[@]}" uv sync --locked --python "$experiment_python" --no-python-downloads
  "${base_run[@]}" uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python "$runtime" --experiment . --record
  flock -s 9
fi
if [[ "${1:-}" == "--matrix" ]]; then
  shift
  exec "${base_run[@]}" uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python src/matrix.py "$@"
fi
exec "${base_run[@]}" uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python "$entrypoint" "$@"
