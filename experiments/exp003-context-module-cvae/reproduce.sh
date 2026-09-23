#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
experiment_python=/home/jy001/micromamba/envs/virtual-cell/bin/python
# Runs share a read lock for their entire lifetime. Only an exclusive holder may
# synchronize the environment, so concurrent trials cannot reinstall each other's
# libraries. The fingerprint is recorded only after a successful locked sync.
environment_key=$(pwd | sha256sum | cut -d ' ' -f 1)
exec 9>"${TMPDIR:-/tmp}/vcc-environment-${environment_key}.lock"
flock -s 9
if ! { test -x .venv/bin/python && micromamba run -n virtual-cell uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python src/runtime.py; }; then
  flock -u 9
  flock -x 9
  micromamba run -n virtual-cell uv sync --locked --python "$experiment_python" --no-python-downloads
  micromamba run -n virtual-cell uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python src/runtime.py --record
  flock -s 9
fi
exec micromamba run -n virtual-cell uv run --locked --no-sync --python "$experiment_python" --no-python-downloads python src/main.py "$@"
