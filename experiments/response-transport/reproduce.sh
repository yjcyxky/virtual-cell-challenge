#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BASE_PYTHON=/home/jy001/micromamba/envs/virtual-cell/bin/python
RUN_ID=transport-mlp-s01
RESUME=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=(--resume); shift ;;
    --run-id) RUN_ID="${2:?--run-id requires a run ID}"; shift 2 ;;
    *) echo "Usage: ./reproduce.sh [--run-id ID] [--resume]" >&2; exit 2 ;;
  esac
done
case "$RUN_ID" in
  transport-mlp-s01)
    CONFIG="configs/$RUN_ID.json" ;;
  *) echo "Unknown run: $RUN_ID" >&2; exit 2 ;;
esac
exec "$BASE_PYTHON" ../../scripts/research.py execute "$RUN_ID" "${RESUME[@]}" -- \
  bash -c 'set -euo pipefail
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    micromamba run -n virtual-cell uv sync --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads
    exec micromamba run -n virtual-cell uv run --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads python -u src/train.py --config "$1"' bash "$CONFIG"
