#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN_ID=local-capability-s02
RESUME=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?--run-id requires ID}"; shift 2 ;;
    --resume) RESUME=(--resume); shift ;;
    *) echo 'Usage: ./reproduce.sh [--run-id ID] [--resume]' >&2; exit 2 ;;
  esac
done
case "$RUN_ID" in
  local-capability-s01|local-capability-s02) CONFIG="configs/$RUN_ID.json" ;;
  *) echo "Unknown registered run: $RUN_ID" >&2; exit 2 ;;
esac
exec /home/jy001/micromamba/envs/virtual-cell/bin/python ../../scripts/research.py execute "$RUN_ID" "${RESUME[@]}" -- \
  bash -c 'set -euo pipefail
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    micromamba run -n virtual-cell uv sync --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads
    exec micromamba run -n virtual-cell uv run --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads python -u src/run.py --config "$1"' bash "$CONFIG"
