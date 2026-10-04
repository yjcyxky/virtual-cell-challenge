#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BASE_PYTHON=/home/jy001/micromamba/envs/virtual-cell/bin/python
export PYTHONPATH="$(pwd)/../../src:$(pwd)/../../scripts:$(pwd)/../../scripts/dossier${PYTHONPATH:+:$PYTHONPATH}"
"$BASE_PYTHON" -m vcc_task.depmap_submission "$@" --gate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
exec micromamba run -n virtual-cell uv run --locked --no-sync --python "$BASE_PYTHON" --no-python-downloads python -u -m vcc_task.depmap_submission "$@"
