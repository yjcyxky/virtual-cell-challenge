#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BASE_PYTHON=/home/jy001/micromamba/envs/virtual-cell/bin/python
RUN_ID=init-linear-s01
RESUME=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=(--resume); shift ;;
    --run-id) RUN_ID="${2:?--run-id requires a run ID}"; shift 2 ;;
    *) echo "Usage: ./reproduce.sh [--run-id ID] [--resume]" >&2; exit 2 ;;
  esac
done
case "$RUN_ID" in
  init-linear-s01) CONFIG=configs/s2-h1-s01.json ;;
  init-linear-nocontext-s01) CONFIG=configs/s2-h1-nocontext-s01.json ;;
  init-linear-nocontext-a01-s01) CONFIG=configs/s2-h1-nocontext-a01-s01.json ;;
  init-program-raw-s01) CONFIG=configs/s2-h1-program-raw-s01.json ;;
  init-pca-s01) CONFIG=configs/s2-h1-pca-s01.json ;;
  init-program-s01) CONFIG=configs/s2-h1-program-s01.json ;;
  init-random-program-s01) CONFIG=configs/s2-h1-random-program-s01.json ;;
  *) echo "Unknown run: $RUN_ID" >&2; exit 2 ;;
esac
exec "$BASE_PYTHON" ../../scripts/research.py execute "$RUN_ID" "${RESUME[@]}" -- \
  bash -c 'set -euo pipefail
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    micromamba run -n virtual-cell uv sync --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads
    exec micromamba run -n virtual-cell uv run --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads python -u src/train.py --config "$1"' bash "$CONFIG"
